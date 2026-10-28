# Simulator guide

The original single-file implementation is retained locally at `archived/simulate.py` (not included in Git). The active entry point is `src/simulate.py`, which calls the `src/simulator/` package. Notebook 01 uses the same package.

## Start with the process

The simulator reads `config/generator.yaml`, then repeats these steps for each customer:

1. Create their profile and habits.
2. Plan their ordinary sessions.
3. Potentially add unusual legitimate activity and a scam episode.
4. Put those sessions in order and record events, applying sending limits.
5. Write the four output CSVs.

Features are calculated afterwards by the existing `features.py`. The simulator does not need existing customer CSVs.

## Files and reading order

| Read | File | Main question |
|---|---|---|
| 1 | `src/simulator/run.py` | What steps happen for every customer, and where are the files saved? |
| 2 | `src/simulator/customers.py` | How does a customer get their group, location, recipients and habits? |
| 3 | `src/simulator/normal_activity.py` | How do those habits produce ordinary sessions and vouchers? |
| 4 | `src/simulator/scenarios.py` | What differs in emergencies, new-phone use, travel and scams? |
| 5 | `src/simulator/events.py` | How are plans turned into dated records and constrained amounts? |
| As needed | `src/simulator/helpers.py` | How do IDs, random locations and positive-value draws work? |
| When running | `src/simulate.py` | How do command-line options load settings and start the run? |

`__init__.py` only exposes `generate_dataset`; it contains no simulation decisions.

## First thing to read in run.py

```python
customer = create_customer(customer_id, config, rng, counters)

sessions = plan_ordinary_activity(customer, config, rng, counters, start_date)
sessions += plan_legitimate_episodes(customer, config, rng, counters)
sessions += plan_scam_episode(customer, config, rng, counters)

record_customer_events(
    customer, sessions, config, rng, counters,
    event_rows, truth_rows, start_date,
)
```

This is the generation process. Start here, then open one called function at a time.

- `config` contains the settings read from YAML.
- `rng` is the seeded random-number generator. Passing the same generator between steps keeps the sequence repeatable.
- `counters` stores the next unique numbers for vouchers, sessions, recipients, devices and episodes.
- `customer` is a dictionary holding this customer's profile and hidden generation habits.
- `sessions` is a list of plans. An amount is still proposed at this point; limits can change it later.
- `event_rows` and `truth_rows` collect the records to write.

## A session plan

For explanation, a simplified plan might look like this:

```python
{
    "start_minute": 600,
    "session_type": "voucher",
    "scenario_type": "normal",
    "episode_id": 0,
    "duration_seconds": 80,
    "screen_count": 3,
    "account_actions": [],
    "vouchers": [
        {"amount": 500, "recipient_id": "P12", "redemption": None}
    ],
}
```

This means a session starting 600 minutes after the observation window begins, with a proposed R500 voucher to P12 and no planned redemption. It is an illustration, not a copied dataset row. In actual plans, redemption is either `None` or a tuple containing delay, location and channel.

The event recorder adds the login, view, recipient-add, voucher and logout records. It also applies daily/monthly room before creating the voucher. It saves the scam label from the scenario name, never from a detection score.

## How the code is organised

Names describe their contents: `customer`, `duration_seconds`, `available_room`, `recipient_id` and `daily_spending`. Most calculations and conditions are split into visible steps. Session plans use descriptive keys, and vouchers use named dictionary entries. Special overrides are named function arguments instead of being extracted from an open-ended `**kw` dictionary.

One small `EventRecorder` class handles repetitive event-row formatting and the current device ID. The process itself is written as ordinary functions. The statistical assumptions are unchanged; comments explain their role without claiming they are real-bank estimates.

## Try a small run first

From the repository root:

```sh
python src/simulate.py --customers 10 --out data/simulator_demo
python src/features.py --data data/simulator_demo
```

The first command creates `customers.csv`, `customers_latent.csv`, `events_log.csv` and `vouchers_truth.csv`. The second creates `cardless_features.csv`. Ten customers are enough for inspecting the file formats, but may produce no scams or no example of a particular scenario.

The command-line default output is `data/simulator_demo`, a separate folder from the existing research dataset. Notebook 01 retains its existing `data/synthetic` output and regeneration switch. Re-running a command into the same output folder replaces that folder's CSVs. Use a new folder name when preserving an earlier demo matters.

The configuration defaults to the existing 20,000 customers if `--customers` is omitted. To run the full original settings in a separate folder:

```sh
python src/simulate.py --out data/simulator_full
python src/features.py --data data/simulator_full
```

## What was preserved deliberately

This is a readability rewrite, not a redesigned experiment. Random draws occur in the original order, including a few redundant draws, so the same seed can reproduce the original records. For example, the original samples a usual hour even when a special scenario supplies an hour; that draw is retained and explained locally.

Other original conventions also remain: screen specifications produce one fewer view event; scam redemption planning has no explicit expiry-discard branch; the CSV's legacy `split` column is still added by the unchanged extractor. Any later change to these would be a separate change to the experiment.

The tests run all 20 dataset consistency checks across several seeds. When the original simulator is available locally, they also compare all four generated CSVs byte for byte and compare the extracted features.

## Current configuration and saved results

The current configuration uses seed 42. The existing dataset also records its original generation settings in `dataset_metadata.json`, including the city labels used for those records. Changing the YAML does not regenerate existing files; regeneration and model refitting are separate operations.
