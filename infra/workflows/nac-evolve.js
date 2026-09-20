export const meta = {
  name: 'nac-evolve',
  description: 'AlphaEvolve + AI co-scientist research loop for one Nordic AI Cup case: islands of different approaches, generate, critique, tournament-rank, execute top ideas, audit, merge, meta-review',
  whenToUse: 'Run once per case (args.case = survival | medical | drone) alongside the track workflows',
  phases: [
    { title: 'Meta-review', detail: 'condense all results and lessons; maintain islands of fundamentally different approaches' },
    { title: 'Generate', detail: 'hypotheses per island, plus a radical island when progress stalls' },
    { title: 'Critique', detail: 'reflection: flaws, overfitting risk, expected gain, cost' },
    { title: 'Rank', detail: 'two-judge tournament, Borda-combined, diversity-preserving selection' },
    { title: 'Execute', detail: 'workers implement top ideas in worktrees and score with the frozen scorer (cascade)' },
    { title: 'Audit', detail: 'hidden-holdout re-score and code review of the best candidate' },
    { title: 'Merge', detail: 'scribe merges, logs, redeploys, queues validation' },
  ],
}

const CASE = args.case
const GENS = args.gens || 4
const POD = args.pod
const EXEC_N = args.execute || 4
const TAG = 'evolve'
const SIGN = 'Commit messages end with:\nCo-Authored-By: Claude Opus 5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_011kJ5oVarFMTmYg5V5ntWDr'

const COMMON = `
TEAM CONTEXT (Nordic AI Cup 2026, team "Phillips"; deadline Sun 20 Sep 16:00 CEST).
- Main repo /home/claude/nac (main). Rules: /home/claude/nac/README.md. Research memory: /home/claude/nac/LEDGER.tsv, /home/claude/nac/<case>/LEDGER.tsv, /home/claude/nac/LESSONS.md, /home/claude/nac/<case>/ISLANDS.md (this workflow's island registry), git branches (git -C /home/claude/nac branch -a; many experiment branches <case>-*).
- Other workflows research the same case in parallel on other tracks; build on their results, don't duplicate them.
- Worker data: /home/claude/upstream-work (hidden holdout excluded). NEVER read /home/claude/.holdout or /workspace/.holdout unless you are the AUDITOR. Never edit scoring files (survival/evaluate.py, medical/offline_eval.py, tools/nac.py, upstream local_evaluator.py/utils.py).
- Pods via python3 /home/claude/nac/infra/pod.py (push <dir> [dest] | exec "<cmd>" <timeout_s> | bg <name> "<cmd>" | get <remote> <local> | url <port>), select with env POD=<gpu|gpu2|gpu3|cpu|cpu2|serve>. THIS WORKFLOW'S COMPUTE: POD=${POD}. Never create/stop/terminate pods; never kill other people's processes (check owners); unique names for pod dirs/jobs.
- Serving must use direct TCP, never the https proxy: survival http://194.68.245.133:22164/predict (POD=serve), drone http://194.68.245.26:22174/predict (POD=gpu port 22), medical: see medical/SHARED_LLM.md and the medical server notes in main.
- Portal: python3 /home/claude/nac/tools/nac.py verify|validate <case> <url> (validation only; the final evaluation is forbidden).
- Never wait more than 5 minutes in one go; long jobs as bg jobs. Cap one experiment at ~35 min wall. If blocked, work around it.
- ${SIGN}
`

const CASES = {
  survival: {
    scorer: 'survival/evaluate.py <policy.py> --seeds train (16 seeds) --procs 8, run on the CPU pod (push worktree, then exec `cd /workspace/<dir>/<basename> && UPSTREAM=/workspace/upstream python3 survival/evaluate.py ...`). Cascade: first --seeds smoke --max-time 600 (sanity), then train. Prints RESULT mean se p25 min deaths=pred/starve/old.',
    facts: 'Rule-based hivemind for herbivores; score ~ seconds until last agent dies (max 3000). v1 = 900.6+-42 (train), validation 1185.6. Mechanics: relative angles; walking 0.05/px, sprint 10x; living 1/s; aging after 60-120 s; child 75 energy, spawn cost 100; predators charge unless faced (or <90 px), rest after bursts; food scarce late (trees die, spawning halves every 300 s); predators grow ~sqrt(t). Other tracks: behaviour/predators, Optuna parameter search, food+breeding.',
  },
  medical: {
    scorer: 'medical/offline_eval.py <preds.json> (--split dev / --split test), preds from medical/pipeline.py; must improve both halves by >2 se. End-to-end timing via upstream local_evaluator.py on the pod (60 s/conversation).',
    facts: 'Audio QA: 10 yes/no questions per consultation -> answers + evidence spans. Score 0.4*acc + 0.6*mean tIoU over true-yes questions. Current: faster-whisper word timestamps -> numbered sentences -> Qwen3.5-35B-A3B GPTQ via vLLM (POD=gpu port 8001, shared) with P(yes) -> quote match -> span. ~0.77 offline on 31 practice conversations; first validation 0.715. Evidence-locator experiment (cross-encoder) 0.629 vs 0.521 in its setting; Qwen3-Reranker being tested by the medical-asr track. The 50/50 yes/no balance holds over the WHOLE set, not per conversation (3-7 yes per conversation) -> global threshold only; yes is +EV above P(yes)~0.23 because tIoU averages over true-yes.',
  },
  drone: {
    scorer: 'Detector: drone/eval_det.py or drone/det_eval.py (frozen leave-block-out detector AP; whichever exists in main or branch drone-alt*). End-to-end: upstream local_evaluator.py (default + --realtime, frames 0-19) against a server on your pod, plus validation runs (249-frame different scene, the most honest signal).',
    facts: 'Small-object detection from a zoomable drone camera (levels 0/1/2 of a 4K frame, 960x540 views, 3 fps, 333 ms budget, answer boxes for the whole frame, one invalid box voids the response). Existing work on branch `drone`: tracker with online ORB+RANSAC homography, greedy coverage planner, server with recorder, YOLO11 v2 trained on ~9.7k synthetic views (weights POD=gpu /root/dd/runs/detect/v2/weights/best.pt; copy to /workspace/drone_weights/). Validation so far 0.089 (old detector); drone-main workflow is deploying v2 now; drone-alt explores template matching and a frozen detector scorer. Latency is decisive (+250 ms -> mAP 0.85 -> 0.44 in realtime sim). Leaderboard top 0.69.',
  },
}
const cfg = CASES[CASE]
const CTX = `${COMMON}\nCASE: ${CASE}\nFACTS: ${cfg.facts}\nFROZEN SCORER: ${cfg.scorer}\n`

const ISLANDS = { type: 'object', properties: {
  lessons: { type: 'array', items: { type: 'string' } },
  dead_ends: { type: 'array', items: { type: 'string' } },
  best: { type: 'object', properties: { ref: { type: 'string' }, score: { type: 'number' }, se: { type: 'number' } }, required: ['ref', 'score', 'se'] },
  stalled: { type: 'boolean' },
  islands: { type: 'array', items: { type: 'object', properties: {
    name: { type: 'string' }, thesis: { type: 'string' }, champion: { type: 'string' }, score: { type: 'number' },
    status: { type: 'string', enum: ['alive', 'stalled', 'retired', 'new'] } },
    required: ['name', 'thesis', 'champion', 'score', 'status'] } } },
  required: ['lessons', 'dead_ends', 'best', 'stalled', 'islands'] }
const HYPS = { type: 'object', properties: { hypotheses: { type: 'array', items: { type: 'object', properties: {
  id: { type: 'string' }, island: { type: 'string' }, title: { type: 'string' }, mechanism: { type: 'string' },
  plan: { type: 'string' }, builds_on: { type: 'string' }, radical: { type: 'boolean' } },
  required: ['id', 'island', 'title', 'mechanism', 'plan', 'builds_on', 'radical'] } } }, required: ['hypotheses'] }
const CRIT = { type: 'object', properties: { reviews: { type: 'array', items: { type: 'object', properties: {
  id: { type: 'string' }, flaws: { type: 'string' }, overfit_risk: { type: 'number' }, expected_gain: { type: 'number' },
  feasibility: { type: 'number' }, novelty: { type: 'number' }, revised_plan: { type: 'string' }, keep: { type: 'boolean' } },
  required: ['id', 'flaws', 'overfit_risk', 'expected_gain', 'feasibility', 'novelty', 'revised_plan', 'keep'] } } }, required: ['reviews'] }
const RANK = { type: 'object', properties: { ranking: { type: 'array', items: { type: 'string' } }, notes: { type: 'string' } }, required: ['ranking', 'notes'] }
const RESULT = { type: 'object', properties: { branch: { type: 'string' }, id: { type: 'string' }, ok: { type: 'boolean' }, score: { type: 'number' }, se: { type: 'number' }, baseline: { type: 'number' }, details: { type: 'string' } }, required: ['branch', 'id', 'ok', 'score', 'se', 'details'] }
const VERDICT = { type: 'object', properties: { pass: { type: 'boolean' }, holdout_score: { type: 'number' }, holdout_baseline: { type: 'number' }, fingerprints_ok: { type: 'boolean' }, reasons: { type: 'string' } }, required: ['pass', 'fingerprints_ok', 'reasons'] }

let history = args.report || ''
let stallCount = 0
let lastBest = null

for (let g = 1; g <= GENS; g++) {
  // ---- Meta-review ----
  const firstGen = g === 1 ? 'Gen 1: design the islands from what has been tried plus at least one approach nobody has tried yet.' : `FOLDED TRACKS: read /home/claude/nac/${CASE}/FOLDED_TRACKS.md. Each folded track is its OWN island (up to 6 islands total) and keeps its separate line of approach; islands share only their best result and these condensed lessons, once per generation. Also read pending handoffs (python3 /home/claude/nac/tools/handoff.py list ${CASE}) and treat each as that island's champion. Rotate: mark islands that did not execute last generation so they get priority.`
  const meta = await agent(`${CTX}\nYou are the ${CASE} META-REVIEWER (co-scientist style), generation ${g}/${GENS}. Read EVERYTHING learned so far for ${CASE}: all ledgers (${CASE}/LEDGER.tsv and ${CASE} rows of LEDGER.tsv), LESSONS.md, ${CASE}/ISLANDS.md if it exists, experiment branches' commit messages and EXPERIMENTS.md files, recent validation results (python3 /home/claude/nac/tools/nac.py status ${CASE}). Then:\n1. Condense lessons: <=12 evidence-backed bullets (each with the numbers), and a dead-ends list (tried and failed, with why). Rewrite the "${CASE}" section of /home/claude/nac/LESSONS.md with them (keep other sections intact), commit in /home/claude/nac main (retry on index.lock).\n2. Maintain 3-4 ISLANDS: fundamentally different approach families for ${CASE} (not parameter tweaks of each other), each with a thesis, its current champion (branch/file) and score. Keep islands alive even if behind, unless clearly dead. ${firstGen} Mark stalled=true if the best score has not improved in the last 2 generations (history below). Write the registry to /home/claude/nac/${CASE}/ISLANDS.md and commit.\nHistory of this workflow:\n${history.slice(-5000)}`,
    { label: `${CASE} meta-review g${g}`, phase: 'Meta-review', schema: ISLANDS, effort: 'high' })
  if (!meta) { log(`g${g}: meta-review failed`); continue }
  const alive = meta.islands.filter(i => i.status !== 'retired').slice(0, 6)
  const radical = meta.stalled || stallCount >= 2
  const L = meta.lessons.join('\n- ')
  const DE = meta.dead_ends.join('\n- ')

  // ---- Generate ----
  const genTasks = alive.map(isl => () => agent(`${CTX}\nYou are a ${CASE} HYPOTHESIS GENERATOR for island "${isl.name}" (thesis: ${isl.thesis}; champion: ${isl.champion}, score ${isl.score}). Condensed lessons:\n- ${L}\nDead ends (don't repeat unless you explain what's different):\n- ${DE}\nCurrent global best: ${meta.best.ref} ${meta.best.score}+-${meta.best.se}.\nPropose 3 hypotheses that push THIS island's approach toward the best TRUE (unseen-data) score. Each must build on specific prior results (say which), explain the causal mechanism, and give a concrete implementation plan runnable in <=35 min. Read code as needed (/home/claude/nac/${CASE}, branches). Use ids like ${isl.name.replace(/\W+/g, '').slice(0, 10)}-g${g}-1.`,
    { label: `${CASE} gen ${isl.name.slice(0, 18)} g${g}`, phase: 'Generate', schema: HYPS }))
  if (radical) {
    genTasks.push(() => agent(`${CTX}\nProgress has STALLED for ${CASE}. You are the RADICAL GENERATOR. Lessons:\n- ${L}\nDead ends:\n- ${DE}\nExisting islands: ${alive.map(i => i.name + ': ' + i.thesis).join(' | ')}\nPropose 3 hypotheses for a NEW island that is fundamentally different from all existing ones (different model family, representation, search method or problem framing); research external methods if useful (web); each testable in <=35 min. Set island to the new island's name, radical=true.`,
      { label: `${CASE} radical g${g}`, phase: 'Generate', schema: HYPS }))
  }
  const genOut = await parallel(genTasks)
  const hyps = genOut.filter(Boolean).flatMap(h => h.hypotheses || [])
  if (!hyps.length) { log(`g${g}: no hypotheses`); continue }

  // ---- Critique ----
  const crit = await agent(`${CTX}\nYou are the ${CASE} REFLECTION reviewer. Critically review each hypothesis below BEFORE any compute is spent: check it against the lessons and dead ends (is it a disguised repeat?), look for flaws, estimate expected gain on the TRUE score (0-1 relative), overfitting/metric-gaming risk (0-1), feasibility in 35 min (0-1), novelty (0-1). Verify claims against code where cheap. Improve the plan where you can (revised_plan). keep=false for repeats, unsound or unsafe ideas.\nLessons:\n- ${L}\nDead ends:\n- ${DE}\nHypotheses:\n${JSON.stringify(hyps, null, 1)}`,
    { label: `${CASE} critique g${g}`, phase: 'Critique', schema: CRIT, effort: 'high' })
  const rv = {}
  if (crit && crit.reviews) crit.reviews.forEach(r => { rv[r.id] = r })
  const kept = hyps.filter(h => !rv[h.id] || rv[h.id].keep)
  if (!kept.length) { log(`g${g}: nothing survived critique`); history += `\nGEN ${g}: all ideas rejected by critique\n`; continue }
  const brief = kept.map(h => ({ id: h.id, island: h.island, title: h.title, mechanism: h.mechanism,
    plan: (rv[h.id] ? rv[h.id].revised_plan : h.plan),
    critique: rv[h.id] ? { flaws: rv[h.id].flaws, gain: rv[h.id].expected_gain, overfit: rv[h.id].overfit_risk, feas: rv[h.id].feasibility, novelty: rv[h.id].novelty } : null }))

  // ---- Rank (two-judge tournament) ----
  const lenses = ['expected improvement of the TRUE hidden-set score per unit of compute', 'robustness: generalises to unseen data, low overfitting/metric-gaming risk, and keeps approach diversity alive']
  const judgeOut = await parallel(lenses.map((lens, i) => () => agent(`${CTX}\nYou are ranking judge ${i + 1} for ${CASE}. Run a pairwise tournament (debate each pair briefly) over these hypotheses using the lens: ${lens}. Return all ids, best first.\n${JSON.stringify(brief, null, 1)}`,
    { label: `${CASE} judge${i + 1} g${g}`, phase: 'Rank', schema: RANK })))
  const ranks = judgeOut.filter(Boolean)
  const pts = {}
  kept.forEach(h => { pts[h.id] = 0 })
  ranks.forEach(r => r.ranking.forEach((id, pos) => { if (id in pts) pts[id] += (kept.length - pos) }))
  const ordered = kept.slice().sort((a, b) => pts[b.id] - pts[a.id])
  const picks = []
  const seenIsl = new Set()
  for (const h of ordered) { if (!seenIsl.has(h.island)) { picks.push(h); seenIsl.add(h.island) } }
  for (const h of ordered) { if (picks.length < EXEC_N && !picks.includes(h)) picks.push(h) }
  const chosen = picks.slice(0, EXEC_N)
  log(`g${g}: ${hyps.length} ideas, ${kept.length} survived critique, executing ${chosen.map(h => h.id).join(', ')}`)

  // ---- Execute ----
  const execOut = await parallel(chosen.map((h, i) => () => agent(`${CTX}\nYou are a ${CASE} WORKER (island ${h.island}). Implement exactly this hypothesis, minimally and cleanly, in your own git worktree on branch ${CASE}-${TAG}-g${g}-${i + 1} (start from main; merge the island champion branch if the plan builds on it):\nID: ${h.id}\nTITLE: ${h.title}\nMECHANISM: ${h.mechanism}\nPLAN: ${rv[h.id] ? rv[h.id].revised_plan : h.plan}\nCRITIQUE TO RESPECT: ${rv[h.id] ? rv[h.id].flaws : 'n/a'}\nCascade: run a cheap sanity check first and abort early if clearly broken or clearly worse; then the full frozen scorer with the same settings as the current best (${meta.best.ref} ${meta.best.score}+-${meta.best.se}); if not comparable, score the best too in the same run. Commit (even if worse). Worktrees can't merge into main. Return JSON (id=${h.id}).`,
    { label: `${CASE} exec g${g}-${i + 1}`, phase: 'Execute', isolation: 'worktree', schema: RESULT })))
  const results = execOut.filter(Boolean)
  const good = results.filter(r => r.ok)
  const base = meta.best.score
  const baseSe = meta.best.se || 0
  const cands = good.filter(r => r.score > ((r.baseline && r.baseline > 0) ? r.baseline : base) + 2 * Math.max(r.se || 0, baseSe)).sort((a, b) => b.score - a.score)

  // ---- Audit ----
  let verdict = null
  if (cands.length) {
    const c = cands[0]
    verdict = await agent(`${CTX}\nYou are the independent AUDITOR (only you may use the hidden holdout; never reveal its contents, report scores only). Candidate ${c.branch} (${c.id}) claims ${c.score}+-${c.se} vs best ${meta.best.ref} ${base}+-${baseSe}. Details: ${c.details}\nRequired: (1) bash /home/claude/.holdout/auditor/audit.sh fingerprints /home/claude/nac all OK and the diff (git -C /home/claude/nac diff main...${c.branch}) touches no scoring files; (2) read the diff: no hardcoded labels/answers/boxes/seeds/file ids, no holdout reads, no special-casing; (3) re-score candidate AND current best on the hidden holdout yourself (survival: POD=cpu, push both policy files, bash /workspace/.holdout/auditor/audit.sh survival <policy.py>; drone: serve each on a spare port, bash /workspace/.holdout/auditor/audit.sh drone <url> on POD=gpu; medical: bash /workspace/.holdout/auditor/audit.sh medical <pod worktree dir> on POD=gpu); (4) pass only if the candidate beats the best on holdout (survival: by >2 se; medical/drone tiny holdout: not worse and consistent with the claimed gain), and practice vs holdout are not wildly apart.`,
      { label: `${CASE} audit g${g}`, phase: 'Audit', schema: VERDICT, effort: 'high' })
  }

  // ---- Merge / log / redeploy ----
  const summary = results.map(r => `${r.branch} [${r.id}]: ok=${r.ok} ${r.score}+-${r.se} (base ${r.baseline}) :: ${String(r.details).slice(0, 400)}`).join('\n')
  const auditTxt = verdict ? JSON.stringify(verdict) : 'no candidate beat the best by >2 se'
  const candTxt = cands.length ? cands[0].branch : 'none'
  const authority = g >= 2 ? `\nYOU ARE THE ${CASE} MERGE AUTHORITY: merges into main and production redeploys need env NAC_MERGER=${CASE} (a git hook and pod.py block everyone else). First process the handoff queue: python3 /home/claude/nac/tools/handoff.py list ${CASE}. For each pending item: if it is unaudited, run the auditor checks yourself (bash /home/claude/.holdout/auditor/audit.sh fingerprints /home/claude/nac; diff review; holdout re-score of candidate vs current best per the audit instructions) and merge only if it passes; if it only syncs production code into main without changing behaviour (e.g. medical-bringup-1), merge it. Record each decision with handoff.py done ${CASE} <branch> merged|rejected "<why>". If you merge a blocked merge by mistake without the env var, run git merge --abort first.` : ''
  const scribe = await agent(`${CTX}${authority}\nYou are the ${CASE} SCRIBE (evolve g${g}). Work in /home/claude/nac main. Results:\n${summary}\nAudit: ${auditTxt}\nCandidate: ${candTxt}\nDo: (a) if the audit passed, merge the candidate into main (retry on index.lock), make it the current best, redeploy the ${CASE} production server from main on its direct-TCP endpoint, and queue a validation in background (nohup python3 /home/claude/nac/tools/nac.py validate ${CASE} <url> > /home/claude/nac/runs/${CASE}_validation_evolve_g${g}.log 2>&1 &). (b) Append one TSV line per result to /home/claude/nac/${CASE}/LEDGER.tsv (agent=evolve, include the island in the idea text; status keep|discard|crash|audit-fail; one-line lesson). (c) Update each island's champion/score in /home/claude/nac/${CASE}/ISLANDS.md. (d) Commit. Return 3 lines: merged?, new best, validation status.`,
    { label: `${CASE} scribe g${g}`, phase: 'Merge', effort: 'low' })
  const newBest = (verdict && verdict.pass) ? cands[0].score : base
  stallCount = (lastBest !== null && newBest <= lastBest) ? stallCount + 1 : 0
  lastBest = Math.max(newBest, lastBest || 0)
  history += `\nGEN ${g}: islands=${alive.map(i => i.name).join(',')} executed=${chosen.map(h => h.id + ':' + h.title).join(' | ')}\n${summary}\nAUDIT=${auditTxt}\nSCRIBE=${scribe}\n`
}
return history.slice(-6000)
