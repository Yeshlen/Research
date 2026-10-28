"""Step 4: turn plans into event rows and voucher truth labels.

This is where sampled amounts meet actual daily/monthly room. There are no
fraud-scoring rules here. A truth label records the originating scenario.
"""
import pandas as pd

from .helpers import MINUTES_PER_DAY, SELF, next_id, round_amount


class EventRecorder:
    """Hold the current device and append one consistently formatted CSV row."""
    def __init__(self, customer_id, device_id, rows):
        self.customer_id = customer_id
        self.device_id = device_id
        self.rows = rows

    def add(
        self, minute, session_id, event_type, voucher_id="", recipient_id="",
        amount="", new_limit="", latitude="", longitude="", channel="",
    ):
        if event_type == "redeem_voucher":
            device_id = ""  # The recipient's redemption is not on the sender's phone.
        else:
            device_id = self.device_id
        if latitude != "":
            latitude = round(latitude, 5)
        if longitude != "":
            longitude = round(longitude, 5)
        self.rows.append((
            self.customer_id, session_id, round(minute, 3), event_type,
            device_id, voucher_id, recipient_id, amount, new_limit,
            latitude, longitude, channel,
        ))


def record_customer_events(customer, sessions, config, rng, counters, event_rows, truth_rows, start_date,
                           initial_state=None):
    product = config["product"]
    window_end = config["days"] * MINUTES_PER_DAY
    # The dashboard can branch from a copied history. Ordinary dataset runs start empty.
    initial_state = initial_state or {}
    current_limit = initial_state.get("current_limit", customer["starting_limit"])
    device_id = f"D{next_id(counters, 'device')}"
    device_id = initial_state.get("device_id", device_id)
    recorder = EventRecorder(customer["customer_id"], device_id, event_rows)
    daily_spending = dict(initial_state.get("daily_spending", {}))
    monthly_spending = dict(initial_state.get("monthly_spending", {}))
    used_recipients = set(initial_state.get("used_recipients", set()))

    sessions_in_order = sorted(sessions, key=lambda session: session["start_minute"])
    for session in sessions_in_order:
        minute = session["start_minute"]
        if minute >= window_end:
            continue
        session_id = next_id(counters, "session")
        recorder.add(minute, session_id, "login")
        minute += rng.uniform(3, 15) / 60
        if session["session_type"] == "background":
            recorder.add(minute, session_id, "view")

        # Record the account changes planned for this session.
        for action in session["account_actions"]:
            minute += rng.uniform(20, 90) / 60
            action_name = action[0]
            if action_name == "device_registered":
                recorder.device_id = f"D{next_id(counters, 'device')}"
                recorder.add(minute, session_id, "device_registered")
            elif action_name == "password_reset":
                recorder.add(minute, session_id, "password_reset")
            elif action_name == "change_limit" and action[1] != current_limit:
                current_limit = action[1]
                recorder.add(minute, session_id, "change_limit", new_limit=current_limit)

        if session["session_type"] == "voucher":
            login_minute = session["start_minute"]
            # Preserve the original convention: screen specification minus one views.
            view_count = max(0, session["screen_count"] - 1)
            for view_index in range(view_count):
                view_minute = (
                    login_minute
                    + (view_index + 1) * session["duration_seconds"] / (view_count + 2) / 60
                )
                recorder.add(view_minute, session_id, "view")
            minute = max(minute, login_minute + session["duration_seconds"] / 60)
            added_in_session = set()

            for voucher_index, voucher in enumerate(session["vouchers"]):
                if voucher_index > 0:
                    minute += rng.uniform(40, 150) / 60
                day = int(minute // MINUTES_PER_DAY)
                month = (start_date + pd.Timedelta(minutes=minute)).month
                daily_room = current_limit - daily_spending.get(day, 0)
                monthly_room = product["monthly_limit"] - monthly_spending.get(month, 0)
                available_room = min(daily_room, monthly_room)
                amount = round_amount(rng, config, voucher["amount"])

                # Keep the condition order: it controls whether a random draw occurs.
                exceeds_room = amount > available_room
                can_raise_limit = current_limit < product["max_daily_limit"]
                legitimate = not session["scenario_type"].startswith("scam")
                if exceeds_room and can_raise_limit and legitimate:
                    if rng.random() < product["p_raise_on_demand"]:
                        current_limit = product["max_daily_limit"]
                        change_minute = minute - rng.uniform(30, 90) / 60
                        recorder.add(change_minute, session_id, "change_limit", new_limit=current_limit)
                        daily_room = current_limit - daily_spending.get(day, 0)
                        monthly_room = product["monthly_limit"] - monthly_spending.get(month, 0)
                        available_room = min(daily_room, monthly_room)

                # Keep only full amount steps within the remaining allowance.
                rounded_room = available_room - available_room % product["amount_step"]
                amount = min(amount, rounded_room)
                if amount < product["amount_step"]:
                    continue  # No voucher if there is not enough room for R50.

                recipient_id = voucher["recipient_id"]
                if (recipient_id != SELF and recipient_id not in used_recipients
                        and recipient_id not in added_in_session):
                    add_minute = minute - rng.uniform(15, 45) / 60
                    recorder.add(add_minute, session_id, "add_recipient", recipient_id=recipient_id)
                    added_in_session.add(recipient_id)

                voucher_id = next_id(counters, "voucher")
                recorder.add(minute, session_id, "create_voucher",
                             voucher_id=voucher_id, recipient_id=recipient_id, amount=amount)
                daily_spending[day] = daily_spending.get(day, 0) + amount
                monthly_spending[month] = monthly_spending.get(month, 0) + amount
                used_recipients.add(recipient_id)

                # Truth comes from the originating scenario, not from a risk score.
                is_scam = int(session["scenario_type"].startswith("scam"))
                truth_rows.append((voucher_id, customer["customer_id"], is_scam,
                                   session["scenario_type"], session["episode_id"]))

                redemption = voucher["redemption"]
                if redemption is not None:
                    delay, location, channel = redemption
                    redemption_minute = minute + delay
                    if redemption_minute < window_end:
                        recorder.add(
                            redemption_minute, session_id, "redeem_voucher",
                            voucher_id=voucher_id, recipient_id=recipient_id, amount=amount,
                            latitude=location[0], longitude=location[1], channel=channel,
                        )
        logout_minute = minute + rng.uniform(10, 60) / 60
        recorder.add(logout_minute, session_id, "logout")
