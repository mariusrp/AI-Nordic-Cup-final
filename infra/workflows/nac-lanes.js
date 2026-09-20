export const meta = {
  name: 'nac-lanes',
  description: 'Fast lane (many cheap tweaks, no judges) or big-bets lane (research + build fundamentally different approaches with full critique) for one Nordic AI Cup case; winners pass confirmation + audit and are handed to the merge authority',
  whenToUse: 'args.case = survival|medical|drone, args.lane = fast|bigbets, args.pod = compute pod',
  phases: [
    { title: 'Plan', detail: 'fast: batches of cheap variants; bigbets: research new approaches' },
    { title: 'Critique', detail: 'bigbets only: reflection + two-judge ranking' },
    { title: 'Build', detail: 'workers implement and score with the frozen scorer' },
    { title: 'Confirm', detail: 'survival: 128 paired fresh seeds; others: both halves / realtime' },
    { title: 'Audit', detail: 'hidden-holdout re-score + code review' },
    { title: 'Handoff', detail: 'submit audited winner to the merge authority, log everything' },
  ],
}

const CASE = args.case
const LANE = args.lane
const POD = args.pod
const ROUNDS = args.rounds || (LANE === 'fast' ? 6 : 1)
const SIGN = 'Commit messages end with:\nCo-Authored-By: Claude Opus 5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_011kJ5oVarFMTmYg5V5ntWDr'

const COMMON = `
TEAM CONTEXT (Nordic AI Cup 2026, team "Phillips"; deadline Sun 20 Sep 16:00 CEST). You are in the ${LANE.toUpperCase()} LANE for ${CASE}; the ${CASE} evolve workflow and the other lane run in parallel.
- Repo /home/claude/nac (main). READ FIRST: /home/claude/nac/LESSONS.md (infrastructure rules + condensed lessons), /home/claude/nac/${CASE}/LEDGER.tsv, /home/claude/nac/${CASE}/ISLANDS.md, /home/claude/nac/${CASE}/FOLDED_TRACKS.md. Never repeat a logged dead end without saying what's different.
- Worker data: /home/claude/upstream-work (hidden holdout excluded). NEVER read /home/claude/.holdout or /workspace/.holdout unless you are the AUDITOR. Never edit scoring files (survival/evaluate.py, medical/offline_eval.py, tools/nac.py, upstream local_evaluator.py/utils.py).
- Pods: python3 /home/claude/nac/infra/pod.py (push <dir> [dest] | exec "<cmd>" <timeout_s> | bg <name> "<cmd>" | get <remote> <local>), env POD=<name>. THIS LANE'S COMPUTE: POD=${POD}. Other pods: gpu = production (medical LLM on :8001 + servers; heavy GPU jobs are blocked there), gpu3 = evolve GPU experiments, gpu4 = big-bets GPU, cpu = survival evolve, cpu2 = survival 128-seed confirmations ONLY, cpu3 = survival lanes, serve = survival production. Never create/stop/terminate pods; never kill other people's processes; unique job/dir names containing your branch.
- You may NOT merge into main or redeploy production (a git hook and pod.py block it). Winners are handed off: python3 /home/claude/nac/tools/handoff.py submit ${CASE} <branch> "<scores, confirmation, audit verdict, summary>". Plain commits of ledger lines to main are allowed.
- Survival rule: a candidate counts only after python3 /home/claude/nac/survival/confirm.py --pod cpu2 --seeds 2000-2127 --baseline v1=main:survival/policies/v1.py --cand <name>=<branch>:<policy.py> says CLEAR WIN (paired, 128 fresh seeds). Validation needs CONFIRMED=<report>.
- Never wait more than 5 minutes in one go (bg jobs + check back). If blocked, work around it. ${SIGN}
`
const SCORER = {
  survival: 'survival/evaluate.py <policy.py> --seeds train --procs 8 on POD=' + POD + ' (cd /workspace/<dir> && UPSTREAM=/workspace/upstream python3 survival/evaluate.py ...); current best v1 ~900 on train (16 seeds), validation 1185.6.',
  medical: 'medical/offline_eval.py <preds.json> --split dev and --split test (must improve both halves by > 2 se); preds from medical/pipeline.py using the shared LLM on POD=gpu :8001 (light clients may run on POD=gpu; heavy jobs on your GPU pod). Current ~0.77 offline, validation 0.715.',
  drone: 'drone/det_eval.py or drone/eval_det.py (frozen leave-block-out detector AP) for detector changes; upstream local_evaluator.py (default + --realtime, frames 0-19) against a server on a spare port of your pod for end-to-end changes. Production drone validation 0.132.',
}[CASE]
const CTX = `${COMMON}\nFROZEN SCORER: ${SCORER}\n`

const RES = { type: 'object', properties: { branch: { type: 'string' }, name: { type: 'string' }, ok: { type: 'boolean' }, score: { type: 'number' }, se: { type: 'number' }, baseline: { type: 'number' }, policy_path: { type: 'string' }, details: { type: 'string' }, next_steps: { type: 'string' } }, required: ['branch', 'name', 'ok', 'score', 'se', 'details'] }
const VERDICT = { type: 'object', properties: { pass: { type: 'boolean' }, reasons: { type: 'string' } }, required: ['pass', 'reasons'] }

async function confirmAuditHandoff(c, label) {
  // Confirm (survival: 128 paired seeds; others: re-run both halves / realtime), then audit, then hand off.
  const conf = await agent(`${CTX}\nCONFIRMATION for candidate ${c.branch} (${c.name}): ${c.score}+-${c.se} vs baseline ${c.baseline}. Details: ${c.details}\n${CASE === 'survival' ? `Run: python3 /home/claude/nac/survival/confirm.py --pod cpu2 --seeds 2000-2127 --tag ${label.replace(/\W+/g, '')} --baseline v1=main:survival/policies/v1.py --cand ${c.name.replace(/\W+/g, '_')}=${c.branch}:${c.policy_path || '<policy path in the branch>'} (it polls; if cpu2 is busy with another confirmation, it just queues). pass=true only if the report verdict is CLEAR WIN.` : CASE === 'medical' ? 'Re-run the candidate and the current best on BOTH offline_eval halves in the same run (fresh run, not cached numbers) and check end-to-end timing < 45 s/conversation. pass=true only if both halves improve by > 2 se and timing is OK.' : 'Re-run upstream local_evaluator.py --realtime for candidate and current production config on the same pod (spare ports), plus det_eval if the detector changed. pass=true only if the candidate is clearly better in realtime (not just offline) and never produces invalid responses.'}`,
    { label: `${label} confirm`, phase: 'Confirm', schema: VERDICT })
  if (!conf || !conf.pass) return { stage: 'confirm', verdict: conf }
  const aud = await agent(`${CTX}\nYou are the independent AUDITOR (only you may use the hidden holdout; report scores only, never contents). Candidate ${c.branch} (${c.name}). Required: (1) bash /home/claude/.holdout/auditor/audit.sh fingerprints /home/claude/nac all OK and git -C /home/claude/nac diff main...${c.branch} touches no scoring files; (2) read the diff: no hardcoded labels/answers/boxes/seeds/file ids, no holdout reads, no special-casing; (3) re-score candidate AND current best on the hidden holdout (survival: POD=cpu3 or cpu, push both policies, bash /workspace/.holdout/auditor/audit.sh survival <policy.py>; drone: serve each on a spare port of POD=gpu, bash /workspace/.holdout/auditor/audit.sh drone <url>; medical: bash /workspace/.holdout/auditor/audit.sh medical <pod worktree dir> on POD=gpu); (4) pass only if the candidate is better on holdout (survival: > 2 se; medical/drone: not worse and consistent with the claimed gain).`,
    { label: `${label} audit`, phase: 'Audit', schema: VERDICT, effort: 'high' })
  if (!aud || !aud.pass) return { stage: 'audit', verdict: aud }
  const h = await agent(`${CTX}\nHand off the audited winner: python3 /home/claude/nac/tools/handoff.py submit ${CASE} ${c.branch} "AUDITED PASS (${LANE} lane): ${c.name} ${c.score}+-${c.se} vs ${c.baseline}; confirmation passed; audit: ${String(aud.reasons).slice(0, 300).replace(/"/g, "'")}". Then append a ledger line to /home/claude/nac/${CASE}/LEDGER.tsv (status handed-off) and commit in /home/claude/nac (plain commit). Return one line.`,
    { label: `${label} handoff`, phase: 'Handoff', effort: 'low' })
  return { stage: 'handed-off', verdict: aud, note: h }
}

let history = args.report || ''

if (LANE === 'fast') {
  const BATCHES = { type: 'object', properties: { best: { type: 'object', properties: { ref: { type: 'string' }, score: { type: 'number' }, se: { type: 'number' } }, required: ['ref', 'score', 'se'] },
    batches: { type: 'array', items: { type: 'object', properties: { title: { type: 'string' }, base: { type: 'string' }, variants: { type: 'array', items: { type: 'string' } } }, required: ['title', 'base', 'variants'] } } }, required: ['best', 'batches'] }
  for (let r = 1; r <= ROUNDS; r++) {
    const plan = await agent(`${CTX}\nYou are the ${CASE} FAST-LANE PLANNER, round ${r}/${ROUNDS}. Propose 2 batches, each with 3-4 CHEAP variants (parameter/threshold tuning, small logic tweaks, ablations) of the current best (or of the most promising island champion), grounded in the lessons and ledger. No big new approaches (the big-bets lane does those). Each variant must be testable within one scorer run. Avoid anything already in the ledger.\nPrevious fast-lane rounds:\n${history.slice(-3000)}`,
      { label: `${CASE} fast plan r${r}`, phase: 'Plan', schema: BATCHES })
    if (!plan) continue
    const out = await parallel(plan.batches.slice(0, 2).map((b, i) => () => agent(`${CTX}\nYou are a ${CASE} FAST-LANE WORKER. In your own git worktree on branch ${CASE}-fast-r${r}-${i + 1} (start from main; merge ${b.base} if it's a branch), implement these variants as separate files/configs and score ALL of them plus the base in ONE scorer run on the same seeds/splits:\nBATCH: ${b.title}\nBASE: ${b.base}\nVARIANTS:\n- ${b.variants.join('\n- ')}\nCommit everything. Return JSON for the BEST variant (name, policy_path for survival, score, se, baseline = base under the same run) and list all variant scores in details.`,
      { label: `${CASE} fast r${r} w${i + 1}`, phase: 'Build', isolation: 'worktree', schema: RES, model: 'sonnet' })))
    const res = out.filter(Boolean).filter(x => x.ok)
    const cands = res.filter(x => x.score > (x.baseline || plan.best.score) + 2 * Math.max(x.se || 0, plan.best.se || 0)).sort((a, b) => b.score - a.score)
    let fate = 'no candidate beat its base by > 2 se'
    if (cands.length) {
      const f = await confirmAuditHandoff(cands[0], `${CASE} fast r${r}`)
      fate = `${cands[0].branch}: ${f.stage} ${f.verdict ? JSON.stringify(f.verdict).slice(0, 300) : ''}`
    }
    const lines = res.map(x => `${x.branch} ${x.name}: ${x.score}+-${x.se} (base ${x.baseline}) :: ${String(x.details).slice(0, 300)}`).join('\n')
    await agent(`${CTX}\nFAST-LANE SCRIBE r${r}: append one TSV line per result below to /home/claude/nac/${CASE}/LEDGER.tsv (agent=fast-lane; status discard | candidate | handed-off per the outcome: ${fate}; one-line lesson each) and commit in /home/claude/nac. Results:\n${lines}`,
      { label: `${CASE} fast scribe r${r}`, phase: 'Handoff', model: 'sonnet', effort: 'low' })
    history += `\nR${r}: ${lines}\nOUTCOME: ${fate}\n`
  }
} else {
  const APPROACHES = { type: 'object', properties: { approaches: { type: 'array', items: { type: 'object', properties: {
    name: { type: 'string' }, thesis: { type: 'string' }, why_it_can_win: { type: 'string' }, evidence: { type: 'string' }, plan: { type: 'string' }, compute: { type: 'string' }, risks: { type: 'string' } },
    required: ['name', 'thesis', 'why_it_can_win', 'evidence', 'plan', 'compute', 'risks'] } } }, required: ['approaches'] }
  const CRIT = { type: 'object', properties: { reviews: { type: 'array', items: { type: 'object', properties: { name: { type: 'string' }, flaws: { type: 'string' }, revised_plan: { type: 'string' }, keep: { type: 'boolean' } }, required: ['name', 'flaws', 'revised_plan', 'keep'] } } }, required: ['reviews'] }
  const RANK = { type: 'object', properties: { ranking: { type: 'array', items: { type: 'string' } }, notes: { type: 'string' } }, required: ['ranking', 'notes'] }
  phase('Plan')
  const research = await agent(`${CTX}\nYou are the ${CASE} BIG-BETS RESEARCHER. Goal: the best TRUE (hidden-set) score. Study what our team already covers (ISLANDS.md, FOLDED_TRACKS.md, ledgers) and research externally (web search, papers, Hugging Face models via the HF tools, leaderboard-style write-ups) for FUNDAMENTALLY DIFFERENT approaches to this problem that could beat our current line by a wide margin. Examples of the kind of difference meant: a learned policy (RL / evolution strategies / imitation) instead of hand rules; a fine-tuned or distilled model instead of prompting; a different detector family or target-domain training instead of synthetic cut-outs. Propose 3 approaches with a concrete milestone plan buildable in ~3 hours on POD=${POD}, the evidence they can work, and risks. ${args.hint || ''}`,
    { label: `${CASE} bigbets research`, phase: 'Plan', schema: APPROACHES, effort: 'high' })
  const crit = await agent(`${CTX}\nYou are the ${CASE} BIG-BETS CRITIC. Review each approach below: fatal flaws, whether it's truly different from what we have, whether it can plausibly beat our best within ~3 hours of building on POD=${POD}, overfitting risk given our tiny labelled data. Improve each plan (revised_plan); keep=false only for unsound ones.\n${JSON.stringify(research ? research.approaches : [], null, 1)}`,
    { label: `${CASE} bigbets critique`, phase: 'Critique', schema: CRIT, effort: 'high' })
  const kept = (research ? research.approaches : []).filter(a => !crit || !crit.reviews.find(r => r.name === a.name && !r.keep))
  const judged = (await parallel(['expected gain on the TRUE score within the remaining time', 'probability of success and robustness to unseen data'].map((lens, i) => () => agent(`${CTX}\nBig-bets judge ${i + 1} (${CASE}). Rank these approaches (names, best first) by: ${lens}. Debate each pair briefly.\n${JSON.stringify(kept.map(a => ({ name: a.name, thesis: a.thesis, plan: (crit && crit.reviews.find(r => r.name === a.name) || {}).revised_plan || a.plan, risks: a.risks })), null, 1)}`,
    { label: `${CASE} bigbets judge${i + 1}`, phase: 'Critique', schema: RANK })))).filter(Boolean)
  const pts = {}
  kept.forEach(a => { pts[a.name] = 0 })
  judged.forEach(j => j.ranking.forEach((n, pos) => { if (n in pts) pts[n] += kept.length - pos }))
  const chosen = kept.slice().sort((a, b) => pts[b.name] - pts[a.name]).slice(0, args.bets || 2)
  log(`big bets chosen: ${chosen.map(a => a.name).join(' | ')}`)
  const ITERS = args.iters || 3
  const finals = await parallel(chosen.map((a, k) => async () => {
    const branch = `${CASE}-bigbet-${k + 1}`
    const plan = (crit && crit.reviews.find(r => r.name === a.name) || {}).revised_plan || a.plan
    let last = null
    for (let it = 1; it <= ITERS; it++) {
      const r = await agent(`${CTX}\nYou are the BIG-BET BUILDER for "${a.name}" (${CASE}), iteration ${it}/${ITERS}.\nTHESIS: ${a.thesis}\nPLAN: ${plan}\nRISKS: ${a.risks}\n${last ? `PREVIOUS ITERATION: ${last.details}\nNEXT STEPS SUGGESTED: ${last.next_steps || ''}` : ''}\nWork in your own git worktree on branch ${branch} (${it === 1 ? 'create it from main' : `continue it: git fetch/checkout ${branch} or merge it into your worktree branch, then commit back onto ${branch}`}). Build toward a working, scorable version fast, then improve. Score with the frozen scorer against the current best in the same run. Budget ~60 min this iteration. Commit. Return JSON (branch=${branch}, name, policy_path for survival, score, se, baseline, details, next_steps).`,
        { label: `${CASE} bet${k + 1} it${it}`, phase: 'Build', isolation: 'worktree', schema: RES })
      if (r) last = r
    }
    return last
  }))
  for (const f of finals.filter(Boolean)) {
    let fate = 'not better than the best'
    if (f.ok && f.score > (f.baseline || 0) + 2 * (f.se || 0)) {
      const o = await confirmAuditHandoff(f, `${CASE} bet ${f.branch}`)
      fate = `${o.stage} ${o.verdict ? JSON.stringify(o.verdict).slice(0, 300) : ''}`
    }
    history += `\n${f.branch} ${f.name}: ${f.score}+-${f.se} (base ${f.baseline}) :: ${String(f.details).slice(0, 500)}\nOUTCOME: ${fate}\n`
  }
  await agent(`${CTX}\nBIG-BETS SCRIBE: write /home/claude/nac/${CASE}/BIGBETS.md (approaches researched, critique, ranking, what was built, scores, outcome, whether each bet should become an evolve island), append ledger lines to /home/claude/nac/${CASE}/LEDGER.tsv (agent=bigbets), and commit in /home/claude/nac.\nChosen: ${chosen.map(a => a.name + ': ' + a.thesis).join(' | ')}\n${history}`,
    { label: `${CASE} bigbets scribe`, phase: 'Handoff', effort: 'low' })
}
return history.slice(-5000)
