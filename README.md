# Agentic AI Investigation Team

An Agentic AI system that investigates enterprise incidents — not just a
network, but evidence pulled from the SaaS tools most enterprises already
use (PagerDuty, ServiceNow, Microsoft Teams, and more as it grows).

This is a fork and generalization of an earlier
[network-specific investigation team](https://github.com/icarovazquez/agentic-ai-network-investigation-team),
rebuilt to treat *any* enterprise data source — network telemetry, vendor
alerts, tickets, application telemetry — as just another input into one
evidence graph.

## Architecture

- **Evidence Graph** — a normalized model of entities and what's been
  observed about them, regardless of which source the evidence came from.
- **Adapters** — one small piece of code per source, translating that
  source's native shape into the graph's normalized shape.
- **Entity Resolver** — shared infrastructure that maps identities and
  locations (a user, a floor) to the network entities they actually
  correspond to.
- **Deterministic Executors & Capabilities** — plain code that answers
  specific questions (reachability, blast radius, traffic state) without
  any LLM involved.
- **Agents** — Incident Framing and the Hypothesis Generator run once.
  Then a loop of Evidence Gap Planner, Evidence Test Creator, deterministic
  execution, Evidence Analyst and Hypothesis Challenger repeats until an
  exit rule in code is met: a supported leader at or above the confidence
  threshold that the Challenger upheld, the same leader for two rounds
  (convergence), a round that returns no new evidence (exhaustion), or the
  round limit. When more than one hypothesis is supported, Causal
  Resolution decides whether they are rivals or a causal chain (an
  underlying cause plus a trigger), and the choice is validated in code.
  Root Cause & Remediation then names the root cause and any contributing
  factors. It recommends; it never acts.
- **Rules live in code, not prompts** — gap caps, id normalization,
  convergence, actionable disputes, grounding checks on LLM extraction and
  entity resolution are enforced by code, because an LLM asked to police
  itself does not reliably do it.

## From notebook to package

This started as one large notebook where every class and agent lived in
cells. It is now a Python package, `investigation_team`, under
`src/investigation_team/` (adapters, agents, capabilities, evidence graph,
orchestrator, evaluation). The notebooks are thin: each one pulls the repo,
installs the package, sets credentials, defines one scenario's
configuration and runs the investigation. The original all-in-one notebook
is still in `notebooks/` for reference.

## Running it

- **Scenarios:** open `notebooks/Scenario 1 - Simple BGP Link Down.ipynb` or
  `notebooks/Scenario 2 - QoS Contention.ipynb` in Colab and run the cells
  top to bottom. After the install cell, restart the runtime once.
- **Credentials:** an Anthropic API key and Langfuse keys, provided as
  environment variables (`ANTHROPIC_API_KEY`, `LANGFUSE_PUBLIC_KEY`,
  `LANGFUSE_SECRET_KEY`). Never commit them.
- **Locally:** `pip install -e .`, then `pytest`. The tests make no LLM
  calls; set the three variables above to any dummy value so the package
  imports.

## Status

This is an active, evolving project, not a finished product. Read the
build log in [this Medium series](https://medium.com/@icaro_vazquez) —
start with
["Agentic AI for the Enterprise"](https://medium.com/@icaro_vazquez/agentic-ai-for-the-enterprise-building-an-investigation-team-that-argues-with-itself-7f68fb06d4fc)
for the architecture, the failure modes, and what's still unfinished.

## Current sources

- NIKA network telemetry (topology + operational state)
- PagerDuty-shaped vendor incidents
- ServiceNow tickets and change records (LLM extraction, checked against
  the ticket text)
- Microsoft Teams call-quality telemetry

## Scenarios

- **Scenario 1: simple BGP link down.** One physical cause, one round
  typically.
- **Scenario 2: QoS contention.** Teams calls degrade although every link is
  up and every ping succeeds. The root cause is a QoS policy detached by an
  overnight upgrade; a backup transfer is only the trigger. Across three
  runs on the final code it was diagnosed correctly each time in one or two
  rounds. Three runs on one scenario is evidence, not a benchmark.

## Known open items

See the "What's Still Unfinished" section of the article above — in short:
the reasoning loop hasn't been stress-tested on genuinely harder,
multi-cause scenarios yet, the evaluation set is one scenario with one
checked outcome, and there's an open question about whether one agent
(Evidence Planner) is being asked to do too much in a single call.
