# Human research decisions and current gate

`ci-design-v1`, 7 September 2026. **NO-GO for headline policy experiments.** The proposed full-information comparison is coherent, but its data feasibility, meaningful margins and independent replication have not been established. No manuscript rewrite or attempt to recover a bandit-positive result is authorized by this design snapshot.

## Decisions requiring human judgment

1. **Target and unit:** ratify unsuccessful first workflow-attempt completion, including timeouts, as the scientific target. This is not a production deployment outcome or one decision per commit. If the intended application is release gating after passing CI, this target must change and different labels are required.
2. **Acquisition route and access:** choose participating repositories with authentic pre-execution event capture, or a trustworthy existing archive. Public retrospective snapshots alone do not satisfy the strict decision contract. Owner cooperation, access rights and pilot resources remain unconfirmed.
3. **Scope of generalization and recruitment:** define the eligible participation frame, language/domain strata, related-repository families and intended population. Decide whether the required sampling breadth is feasible; otherwise explicitly plan a bounded case study.
4. **Missingness and capture tradeoff:** ratify the proposed pre-execution capture/diff-coverage pilot thresholds and 30-day follow-up. Accept that conditional-on-resolved results and capture attrition limit conclusions, or fund an acquisition method that improves coverage. Do not turn canceled runs into failures merely to improve sample size.
5. **Scientific importance margins:** ratify or replace proposed normalized-cost .025 and Brier .01 margins before evaluation. Their meaning is under hypothetical costs; only practitioners/application evidence could establish operational significance. These choices drive precision needs.
6. **Resource/precision budget:** fund the independent projects and fixed observation period required by the prospective plan. The 5/5/10 split is only a starting budget. Smaller effects/greater heterogeneity may require many more clusters; if infeasible, choose exploratory claims before seeing results.
7. **Feature interpretation:** ratify event-specific diff basis, language-biased dependency/test/configuration patterns, workflow categories and optional author/repository metadata. Nulls and uncertain identity must remain honest. A richer collector is not guaranteed to produce predictive context.
8. **Custody and holdout governance:** designate who controls sealed evaluation data, what aggregate quality monitoring is permissible, where the protocol is time-stamped/registered, and how deviations invalidate or amend claims. A local hash cannot prove that no one looked at public outcomes.
9. **Data rights and identities:** decide raw-data redistribution, provider identity versus pseudonymous author keys, and access/retention obligations for participating repositories.
10. **Final paper scope:** current preferred framing is cost-sensitive CI risk prediction with delayed full-information feedback. Bandits remain comparators. A bandit-primary or real deployment-control paper requires additional application/feedback evidence, not a cosmetic title change.

## What would change the decision to GO

Ratified target/cost/outcome/margin decisions; demonstrated development-pilot capture and meaningful feature coverage; authentic source lineage and all required integration tests; independently locked project partitions and fixed dates; a precision plan supported by justified external evidence or conservative variance assumptions (with any authorized development refinement frozen before evaluation access); fresh-environment reproduction; and human approval of the exact hash-bound protocol/runner/data versions before holdout access.

GO would mean that the acquired/proposed data and approved protocol can answer the **scoped** research question, allowing inconclusive and negative outcomes. It would not promise a significant effect, make the project submission-ready, or permit production-benefit language. Until then, only specification, fixture testing and separately authorized data-only feasibility work should proceed.
