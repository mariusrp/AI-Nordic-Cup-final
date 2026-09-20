export const meta = {
  name: 'nac-understand',
  description: 'Understanding lane for one Nordic AI Cup case: a strong agent repeatedly studies the rules, scoring code, simulator/data and our errors, verifies findings with quick measurements, and feeds them to the other lanes via LESSONS.md',
  whenToUse: 'args.case = survival|medical|drone, args.pod = pod for quick measurements, args.cycles = number of study cycles',
  phases: [{ title: 'Study', detail: 'deep study of rules, scoring, data and our errors; verified findings' }, { title: 'Publish', detail: 'findings written to LESSONS.md for the other lanes' }],
}

const CASE = args.case
const POD = args.pod
const CYCLES = args.cycles || 4
const FIND = { type: 'object', properties: {
  findings: { type: 'array', items: { type: 'object', properties: {
    title: { type: 'string' }, evidence: { type: 'string', description: 'code references and the numbers you measured' },
    points_at_stake: { type: 'string', description: 'estimated score impact and why' },
    action: { type: 'string', description: 'what a lane should build or test' }, lane: { type: 'string', enum: ['fast', 'bigbets', 'evolve', 'all'] },
    confidence: { type: 'string', enum: ['verified', 'likely', 'hypothesis'] } },
    required: ['title', 'evidence', 'points_at_stake', 'action', 'lane', 'confidence'] } },
  loss_breakdown: { type: 'string', description: 'where our current solution loses points, quantified' },
  next_questions: { type: 'array', items: { type: 'string' } } },
  required: ['findings', 'loss_breakdown', 'next_questions'] }

const FOCUS = {
  survival: 'Simulator source: /home/claude/upstream-work/survival-simulator/src (environment.py, creature.py, predator.py, agent.py, biome.py, sensing.py, simulation.py) plus the organisers\' server contract (DTOs, agent_server.py; note the organiser sends lowercase observation types). Score = time until the last agent dies + fruit/1000 - eaten energy/100. Study: exact predator state machine (rest/wake thresholds, energy costs, pivot when faced, charge radius), spawn rules (tree/fruit/predator over time: measure it), aging, sprint limits, biome modifiers, river flow, what kills our agents and WHEN (run v1 and the top candidates on seeds 2000+ with instrumentation on POD=' + POD + '), and what a near-3000 game looks like. Check Adrian\'s claims: food never runs out (~5 trees at 2500 s), predators grow to ~20.',
  medical: 'Upstream medical-appointment/: README, local_evaluator.py/utils.py (exact scoring: accuracy + tIoU, how nulls count), data/question_train.csv with the gold evidence spans (only the 31 visible conversations), audio. Study how annotators choose spans (starts vs ends, relation to speech onsets, pauses, clause/sentence boundaries, which mention when repeated, question types), how question wording relates to the evidence, where OUR pipeline loses tIoU per question (use cached predictions in medical/cache, runs on POD=gpu /workspace/runs/med), and the validation vs practice gap (0.72 vs 0.78).',
  drone: 'Upstream drone-flyby/: README, local_evaluator.py/utils.py (exact COCO mAP@0.5 settings: maxDets, area ranges, per-class averaging, how absent classes and duplicates are treated), dtos (camera constraints), the Helsinki data, and our recorded validation views on POD=gpu /workspace/drone_seen (frames <= 150 only; never frames >= 151). Study where our mAP is lost: per class, per zoom level, recall vs precision, box size/IoU errors, camera coverage (which objects are never looked at closely), latency and skipped frames, and how scoring rewards many honest candidates vs precision-first reporting.',
}[CASE]

let history = ''
for (let c = 1; c <= CYCLES; c++) {
  const r = await agent(`You are the ${CASE.toUpperCase()} UNDERSTANDING LANE (Nordic AI Cup 2026, team "Phillips"; deadline Sun 20 Sep 16:00 CEST), study cycle ${c}/${CYCLES}. Other lanes (evolve, fast, big bets) test ideas; your job is to deeply UNDERSTAND the game so they test the right things. Winners of Ambolt's earlier competitions credit simple proven methods and exploring different strategies early.

Read first: /home/claude/nac/LESSONS.md (incl. PRIORITY DIRECTIVES), /home/claude/nac/${CASE}/LEDGER.tsv, ${CASE}/ISLANDS.md, ${CASE}/FOLDED_TRACKS.md, ${CASE}/BIGBETS.md, the current production code in /home/claude/nac/${CASE}.
FOCUS: ${FOCUS}

Find where we lose points and what we're not exploiting. Prefer facts you can prove from the code and quick measurements (instrumented runs, error breakdowns, small analysis scripts) over opinions. Quantify points at stake. Also re-check lessons that might be wrong. ${c > 1 ? 'Start by reading what changed since your last cycle (new ledger lines, validation scores, merges) and follow up your open questions.' : ''}

Rules: do NOT change policies, models, servers or scoring files; analysis scripts go in /home/claude/nac/${CASE}/analysis/ (commit them). Worker data is /home/claude/upstream-work; NEVER read /home/claude/.holdout or /workspace/.holdout, never use survival seeds 5000-5031, drone recorded frames >= 151 or the 8 hidden medical conversations. Compute for measurements: python3 /home/claude/nac/infra/pod.py with env POD=${POD} (push/exec/bg/get); never create/stop pods, never kill others' processes, never wait > 5 min in one go. Budget ~45 minutes for this cycle.

PUBLISH: write your verified and likely findings into /home/claude/nac/LESSONS.md under a section "## UNDERSTANDING: ${CASE} (cycle N, time)" placed right after the PRIORITY DIRECTIVES section (replace your previous cycle's section; keep everything else intact), each finding with evidence, points at stake and which lane should act. Commit in /home/claude/nac (plain commit; retry on index.lock; commit messages end with:
Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_011kJ5oVarFMTmYg5V5ntWDr). Return JSON.
${history ? `\nYour previous cycles:\n${history.slice(-4000)}` : ''}`,
    { label: `${CASE} understand c${c}`, phase: 'Study', schema: FIND, effort: 'max' })
  if (!r) { log(`c${c}: no result`); continue }
  log(`c${c}: ${r.findings.length} findings (${r.findings.filter(f => f.confidence === 'verified').length} verified)`)
  history += `\nCYCLE ${c}: loss breakdown: ${r.loss_breakdown}\nFINDINGS: ${r.findings.map(f => `[${f.confidence}/${f.lane}] ${f.title}: ${f.points_at_stake}`).join(' | ')}\nOPEN QUESTIONS: ${r.next_questions.join(' | ')}\n`
}
return history.slice(-6000)
