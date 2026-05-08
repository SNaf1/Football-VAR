import {
  startTransition,
  useDeferredValue,
  useEffect,
  useMemo,
  useRef,
  useState,
  type MouseEvent as ReactMouseEvent,
} from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import {
  BrowserRouter,
  NavLink,
  Navigate,
  Route,
  Routes,
  useLocation,
  useNavigate,
  useParams,
} from 'react-router-dom'
import clsx from 'clsx'

import {
  applyOffsideCorrection,
  createMatch,
  createReviewClip,
  deleteIncident,
  demoLogin,
  getIncident,
  getIncidents,
  getSampleClips,
  healthcheck,
  reviewGoal,
  reviewOffsideFrame,
  saveIncidentNote,
  toAssetUrl,
  uploadVideo,
} from './lib/api'
import { usePersistentState } from './lib/storage'
import type {
  AttackDirection,
  DemoSession,
  DraftMatch,
  IncidentRecord,
  MatchRecord,
  PlayerCandidate,
  ReviewClip,
  ReviewType,
  Role,
  VideoAsset,
} from './types'

type ExpandedMedia = {
  src: string
  title: string
  alt: string
  version?: string | null
}

type CorrectionTarget = 'attacker' | 'defender'

const buttonStyles =
  'display-face inline-flex items-center justify-center gap-3 rounded-none border border-white/12 px-4 py-3 text-base font-semibold uppercase tracking-[0.18em] transition duration-200 hover:border-sky-200/30 hover:bg-white/8 focus:outline-none focus:ring-2 focus:ring-sky-300/40'

const subtleButtonStyles =
  'inline-flex items-center justify-center rounded-none border border-white/10 bg-white/[0.04] px-3 py-2 text-[0.72rem] font-medium uppercase tracking-[0.16em] text-slate-100/92 transition hover:border-white/20 hover:bg-white/[0.08]'

function App() {
  const [session, setSession] = usePersistentState<DemoSession | null>('ai_session', null)
  const [draftMatch, setDraftMatch] = usePersistentState<DraftMatch | null>('ai_draft_match', null)
  const [activeMatch, setActiveMatch] = usePersistentState<MatchRecord | null>('ai_active_match', null)
  const [activeVideo, setActiveVideo] = usePersistentState<VideoAsset | null>('ai_active_video', null)
  const [incidentCache, setIncidentCache] = usePersistentState<Record<string, IncidentRecord>>('ai_incident_cache', {})

  return (
    <BrowserRouter>
      <Routes>
        <Route
          path="/"
          element={
            <LoginPage
              session={session}
              onLogin={(nextSession) => {
                setSession(nextSession)
              }}
            />
          }
        />
        <Route
          path="/*"
          element={
            session ? (
              <ProtectedApp
                session={session}
                draftMatch={draftMatch}
                activeMatch={activeMatch}
                activeVideo={activeVideo}
                incidentCache={incidentCache}
                setSession={setSession}
                setDraftMatch={setDraftMatch}
                setActiveMatch={setActiveMatch}
                setActiveVideo={setActiveVideo}
                setIncidentCache={setIncidentCache}
              />
            ) : (
              <Navigate to="/" replace />
            )
          }
        />
      </Routes>
    </BrowserRouter>
  )
}

interface ProtectedAppProps {
  session: DemoSession
  draftMatch: DraftMatch | null
  activeMatch: MatchRecord | null
  activeVideo: VideoAsset | null
  incidentCache: Record<string, IncidentRecord>
  setSession: (value: DemoSession | null) => void
  setDraftMatch: (value: DraftMatch | null) => void
  setActiveMatch: (value: MatchRecord | null) => void
  setActiveVideo: (value: VideoAsset | null) => void
  setIncidentCache: (value: Record<string, IncidentRecord>) => void
}

function ProtectedApp(props: ProtectedAppProps) {
  const location = useLocation()
  const sidebarNavItems =
    props.session.role === 'Team Viewer'
      ? [{ to: '/viewer', label: 'Team Viewer' }]
      : [
          { to: '/match/new', label: 'Create Match' },
          { to: '/video', label: 'Load Video' },
          { to: '/console', label: 'Match Console' },
          { to: '/incidents', label: 'Incident Log' },
        ]

  useEffect(() => {
    window.scrollTo(0, 0)
  }, [location.pathname])

  return (
    <div className="min-h-screen text-slate-50 lg:flex">
      <aside className="panel-cut hidden h-screen w-[18.5rem] shrink-0 overflow-y-auto border-r border-white/8 p-5 lg:sticky lg:top-0 lg:flex lg:flex-col lg:justify-between">
        <div className="min-h-0">
          <div className="mb-8">
            <p className="display-face text-[0.72rem] uppercase tracking-[0.32em] text-sky-100/70">Atletico Intelligence</p>
            <h1 className="display-face mt-2 text-5xl font-bold uppercase tracking-[0.18em]">Review Hub</h1>
          </div>
          <div className="mb-8 panel-cut pitch-grid p-4">
            <p className="display-face text-xs uppercase tracking-[0.3em] text-slate-300/70">Signed In</p>
            <p className="mt-3 text-2xl font-semibold">{props.session.display_name}</p>
            <p className="mt-1 text-sm text-slate-400">{props.session.role}</p>
          </div>
          <nav className="space-y-2">
            {sidebarNavItems.map((item) => (
              <AppNavItem key={item.to} to={item.to} label={item.label} />
            ))}
          </nav>
        </div>

        <div className="space-y-3">
          <div className="panel-cut p-4">
            <p className="display-face text-xs uppercase tracking-[0.28em] text-slate-300/70">Current Surface</p>
            <p className="mt-2 text-lg font-semibold">{props.activeMatch?.title ?? 'No live match prepared'}</p>
            <p className="muted-copy mt-1 text-sm">
              {props.activeVideo ? `${props.activeVideo.name} · ${formatSeconds(props.activeVideo.duration)}` : 'Load a clip to arm the console.'}
            </p>
          </div>
          <button
            className={clsx(buttonStyles, 'w-full bg-slate-50 text-slate-950 hover:bg-slate-100')}
            onClick={() => {
              props.setSession(null)
              props.setDraftMatch(null)
              props.setActiveMatch(null)
              props.setActiveVideo(null)
            }}
            type="button"
          >
            Sign Out
          </button>
        </div>
      </aside>

      <main className="min-h-screen min-w-0 flex-1 px-4 py-4 md:px-5 lg:px-6">
        <div className="panel-cut mb-4 flex flex-col gap-4 px-5 py-4 md:flex-row md:items-center md:justify-between">
          <div>
            <p className="display-face text-[0.7rem] uppercase tracking-[0.34em] text-sky-100/70">Match Operations</p>
            <h2 className="surface-heading mt-2 text-[2.45rem] font-semibold tracking-[-0.05em] text-slate-50 md:text-[2.9rem]">
              {headlineForPath(location.pathname)}
            </h2>
          </div>
          <div className="grid gap-3 md:grid-cols-3">
            <MetricCell label="Role" value={props.session.role} />
            <MetricCell label="Match" value={props.activeMatch ? `${props.activeMatch.home_team} vs ${props.activeMatch.away_team}` : 'Ready'} />
            <MetricCell label="Incidents" value={String(Object.keys(props.incidentCache).length).padStart(2, '0')} />
          </div>
        </div>

        <AnimatePresence mode="wait">
          <Routes>
            <Route
              path="/match/new"
              element={<CreateMatchPage draftMatch={props.draftMatch} setDraftMatch={props.setDraftMatch} role={props.session.role} />}
            />
            <Route
              path="/video"
              element={
                <LoadVideoPage
                  draftMatch={props.draftMatch}
                  setActiveMatch={props.setActiveMatch}
                  setActiveVideo={props.setActiveVideo}
                  role={props.session.role}
                />
              }
            />
            <Route
              path="/console"
              element={
                <MatchConsolePage
                  activeMatch={props.activeMatch}
                  activeVideo={props.activeVideo}
                  role={props.session.role}
                  onIncident={(incident) => {
                    props.setIncidentCache({ ...props.incidentCache, [incident.id]: incident })
                  }}
                />
              }
            />
            <Route
              path="/incidents"
              element={
                <IncidentLogPage
                  role={props.session.role}
                  incidentCache={props.incidentCache}
                  setIncidentCache={props.setIncidentCache}
                />
              }
            />
            <Route
              path="/incidents/:incidentId"
              element={
                <IncidentDetailPage
                  role={props.session.role}
                  incidentCache={props.incidentCache}
                  setIncidentCache={props.setIncidentCache}
                />
              }
            />
            <Route
              path="/viewer"
              element={
                <TeamViewerPage
                  incidentCache={props.incidentCache}
                  setIncidentCache={props.setIncidentCache}
                />
              }
            />
            <Route path="*" element={<Navigate to={props.session.role === 'Team Viewer' ? '/viewer' : '/match/new'} replace />} />
          </Routes>
        </AnimatePresence>
      </main>
    </div>
  )
}

function LoginPage({
  session,
  onLogin,
}: {
  session: DemoSession | null
  onLogin: (session: DemoSession) => void
}) {
  const navigate = useNavigate()
  const [role, setRole] = useState<Role>('Match Official')
  const [displayName, setDisplayName] = useState('Sam Rivera')
  const [email, setEmail] = useState('official@league.com')
  const [loginOpen, setLoginOpen] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [healthState, setHealthState] = useState<'checking' | 'online' | 'offline'>('checking')
  const [error, setError] = useState('')

  useEffect(() => {
    if (session) {
      navigate(session.role === 'Team Viewer' ? '/viewer' : '/match/new', { replace: true })
    }
  }, [navigate, session])

  useEffect(() => {
    healthcheck()
      .then(() => setHealthState('online'))
      .catch(() => setHealthState('offline'))
  }, [])

  const submit = async () => {
    setSubmitting(true)
    setError('')
    try {
      const nextSession = await demoLogin(role, displayName, email)
      startTransition(() => {
        onLogin(nextSession)
        navigate(role === 'Team Viewer' ? '/viewer' : '/match/new', { replace: true })
      })
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : 'Sign-in failed.')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <PageFrame className="grid min-h-screen place-items-center px-4 py-6">
      <motion.section
        initial={{ opacity: 0, y: 18 }}
        animate={{ opacity: 1, y: 0 }}
        className="panel-cut login-hero w-full max-w-[1380px] overflow-hidden px-6 py-8 md:px-8 lg:px-10"
      >
        <div className="flex items-center gap-3">
          <FootballMark className="h-8 w-8 text-sky-100/92" />
          <p className="display-face text-[0.78rem] uppercase tracking-[0.34em] text-sky-100/70">Single-Camera Incident Review</p>
        </div>
        <div className="kicker-line mt-4" />
        <div className="mt-8 grid gap-10 xl:grid-cols-[1.16fr_0.84fr] xl:items-center">
          <div>
            <h1 className="surface-heading max-w-[10ch] text-[3.45rem] text-slate-50 md:text-[4.55rem] xl:text-[4.85rem]">
              Lock the frame. Read the line. Call the moment.
            </h1>
            <p className="muted-copy mt-6 max-w-[56ch] text-base leading-7">
              One clear football review surface for offside and goal-line decisions, built around frame lock, line projection, and a clean incident archive.
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              <span className="review-badge text-sky-100/88">Frame Lock</span>
              <span className="review-badge text-emerald-100/88">Line Projection</span>
              <span className="review-badge text-amber-100/88">Incident Archive</span>
            </div>
            <div className="mt-8 flex flex-wrap gap-3">
              <button
                className={clsx(buttonStyles, 'bg-slate-50 px-5 text-slate-950 hover:bg-slate-100')}
                type="button"
                onClick={() => setLoginOpen(true)}
              >
                Demo Login
              </button>
              <span className="muted-copy self-center text-sm">Open the review console in a modal and choose your role.</span>
            </div>
          </div>
          <div className="xl:justify-self-end">
            <FootballHeroGraphic />
          </div>
        </div>

        <div className="mt-10 grid gap-4 lg:grid-cols-3">
          <TutorialStep
            index="01"
            title="Create the match"
            body="Set the fixture name, kickoff, and operator role before any clip is loaded."
          />
          <TutorialStep
            index="02"
            title="Load and lock the moment"
            body="Upload a clip or pick a sample, then trim the incident and scrub to the pass or goal frame."
          />
          <TutorialStep
            index="03"
            title="Review and archive"
            body="Check the line, add a referee note, and log the incident for the viewer archive."
          />
        </div>
      </motion.section>

      <AnimatePresence>
        {loginOpen ? (
          <motion.div
            className="fixed inset-0 z-[90] flex items-center justify-center bg-slate-950/88 px-4 py-6 backdrop-blur-md"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setLoginOpen(false)}
          >
            <motion.section
              className="panel-cut w-full max-w-[720px] px-6 py-8 md:px-8"
              initial={{ opacity: 0, y: 20, scale: 0.985 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 14, scale: 0.985 }}
              transition={{ duration: 0.22, ease: 'easeOut' }}
              onClick={(event) => event.stopPropagation()}
            >
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="display-face text-[0.78rem] uppercase tracking-[0.34em] text-sky-100/70">Demo Access</p>
                  <h2 className="surface-heading mt-3 text-[2.4rem] text-slate-50">Enter the console</h2>
                </div>
                <button className={clsx(subtleButtonStyles, 'px-4 py-2')} type="button" onClick={() => setLoginOpen(false)}>
                  Close
                </button>
              </div>

              <div className="mt-8 space-y-6">
                <div>
                  <label className="display-face text-[0.72rem] uppercase tracking-[0.28em] text-slate-300/80">Role</label>
                  <div className="mt-3 grid gap-3 sm:grid-cols-2">
                    {(['Match Official', 'Team Viewer'] as const).map((option) => (
                      <button
                        key={option}
                        className={clsx(buttonStyles, 'px-3 py-3 text-sm', role === option ? 'border-sky-200/50 bg-sky-300/14 text-white' : 'text-slate-200')}
                        type="button"
                        onClick={() => setRole(option)}
                      >
                        {option}
                      </button>
                    ))}
                  </div>
                </div>

                <Field label="Display Name" value={displayName} onChange={setDisplayName} placeholder="Sam Rivera" />
                <Field label="Email" value={email} onChange={setEmail} placeholder="official@league.com" />

                {healthState === 'offline' ? (
                  <div className="rounded-none border border-amber-300/20 bg-amber-300/10 px-4 py-3 text-sm text-amber-100">
                    Backend is offline right now. Start the local API before signing in.
                  </div>
                ) : null}
                {error ? <div className="rounded-none border border-rose-300/20 bg-rose-300/10 px-4 py-3 text-sm text-rose-100">{error}</div> : null}

                <button
                  className={clsx(buttonStyles, 'w-full bg-slate-50 text-slate-950 hover:bg-slate-100 disabled:cursor-wait disabled:opacity-70')}
                  onClick={submit}
                  disabled={submitting}
                  type="button"
                >
                  {submitting ? 'Signing In...' : 'Open Review Surface'}
                </button>
              </div>
            </motion.section>
          </motion.div>
        ) : null}
      </AnimatePresence>
    </PageFrame>
  )
}

function CreateMatchPage({
  draftMatch,
  setDraftMatch,
  role,
}: {
  draftMatch: DraftMatch | null
  setDraftMatch: (value: DraftMatch) => void
  role: Role
}) {
  const navigate = useNavigate()
  const [title, setTitle] = useState(draftMatch?.title ?? 'PSG vs Bayern Munich')
  const [homeTeam, setHomeTeam] = useState(draftMatch?.homeTeam ?? 'PSG')
  const [awayTeam, setAwayTeam] = useState(draftMatch?.awayTeam ?? 'Bayern Munich')
  const [kickoff, setKickoff] = useState(draftMatch?.kickoff ?? '2026-05-06T18:00')

  if (role === 'Team Viewer') {
    return <Navigate to="/viewer" replace />
  }

  return (
    <PageFrame key="match-new">
      <motion.section initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="grid gap-6 xl:grid-cols-[1fr_0.86fr]">
        <div className="panel-cut p-6 md:p-8">
          <p className="display-face text-[0.72rem] uppercase tracking-[0.32em] text-sky-100/70">Workflow Step 01</p>
          <h3 className="surface-heading mt-3 text-[2.85rem] text-slate-50">Create the match card</h3>
          <p className="muted-copy mt-4 max-w-[52ch] text-base leading-7">Set the fixture and kickoff, then attach the clip on the next screen.</p>
          <div className="mt-8 grid gap-5 md:grid-cols-2">
            <Field label="Home Team" value={homeTeam} onChange={setHomeTeam} placeholder="PSG" />
            <Field label="Away Team" value={awayTeam} onChange={setAwayTeam} placeholder="Bayern Munich" />
            <div className="md:col-span-2">
              <Field label="Match Title" value={title} onChange={setTitle} placeholder="PSG vs Bayern Munich" />
            </div>
            <div className="md:col-span-2">
              <Field label="Kickoff" value={kickoff} onChange={setKickoff} placeholder="2026-05-06T18:00" type="datetime-local" />
            </div>
          </div>
        </div>

        <div className="space-y-6">
          <div className="panel-cut p-6">
            <p className="display-face text-[0.72rem] uppercase tracking-[0.32em] text-sky-100/70">Operator Notes</p>
            <div className="mt-5 grid gap-4">
              <MetricNarrative title="One active review" body="Keep a single incident open while locking the decision frame." />
              <MetricNarrative title="Clip archive" body="Each review stores the incident window and generated decision assets." />
              <MetricNarrative title="Team viewer" body="Incidents remain visible in read-only mode after they are logged." />
            </div>
          </div>
          <button
            className={clsx(buttonStyles, 'w-full bg-emerald-300/90 text-slate-950 hover:bg-emerald-200')}
            type="button"
            onClick={() => {
              setDraftMatch({ title, homeTeam, awayTeam, kickoff })
              navigate('/video')
            }}
          >
            Continue to Load Video
          </button>
        </div>
      </motion.section>
    </PageFrame>
  )
}

function LoadVideoPage({
  draftMatch,
  setActiveMatch,
  setActiveVideo,
  role,
}: {
  draftMatch: DraftMatch | null
  setActiveMatch: (value: MatchRecord | null) => void
  setActiveVideo: (value: VideoAsset | null) => void
  role: Role
}) {
  const navigate = useNavigate()
  const [samples, setSamples] = useState<VideoAsset[]>([])
  const [selectedVideo, setSelectedVideo] = useState<VideoAsset | null>(null)
  const [loadingSamples, setLoadingSamples] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    getSampleClips()
      .then(setSamples)
      .catch(() => setSamples([]))
      .finally(() => setLoadingSamples(false))
  }, [])

  if (role === 'Team Viewer') {
    return <Navigate to="/viewer" replace />
  }

  if (!draftMatch) {
    return <Navigate to="/match/new" replace />
  }

  const handleUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return
    setUploading(true)
    setError('')
    try {
      const uploaded = await uploadVideo(file)
      setSelectedVideo(uploaded)
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : 'Upload failed.')
    } finally {
      setUploading(false)
    }
  }

  const createLinkedMatch = async () => {
    if (!selectedVideo) {
      setError('Choose a sample clip or upload a match segment first.')
      return
    }
    setSubmitting(true)
    setError('')
    try {
      const nextMatch = await createMatch({
        title: draftMatch.title,
        home_team: draftMatch.homeTeam,
        away_team: draftMatch.awayTeam,
        kickoff: draftMatch.kickoff,
        video_id: selectedVideo.id,
      })
      startTransition(() => {
        setActiveMatch(nextMatch)
        setActiveVideo(selectedVideo)
        navigate('/console')
      })
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : 'Could not create the match.')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <PageFrame key="video-load">
      <motion.section initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="grid gap-6 xl:grid-cols-[1.06fr_0.94fr]">
        <div className="panel-cut p-6 md:p-8">
          <p className="display-face text-[0.72rem] uppercase tracking-[0.32em] text-sky-100/70">Workflow Step 02</p>
          <h3 className="surface-heading mt-3 text-[2.85rem] text-slate-50">Load the match feed</h3>
          <p className="muted-copy mt-4 max-w-[60ch] text-base leading-7">
            Use bundled sample clips for a predictable demo, or upload a short MP4 to test your own incident footage locally. The console will trim from this source when the operator hits a review button.
          </p>

          <label className="panel-cut mt-8 block cursor-pointer border-dashed border-sky-100/18 bg-sky-300/6 p-6 transition hover:border-sky-100/30 hover:bg-sky-300/10">
            <span className="display-face text-sm uppercase tracking-[0.28em] text-sky-100/70">Upload Match Segment</span>
            <p className="mt-4 text-xl font-semibold">Choose a local MP4, MOV, MKV, AVI, or WEBM file</p>
            <p className="muted-copy mt-2 text-sm">Uploads stay local to the backend storage directory during testing.</p>
            <input type="file" accept="video/*" className="hidden" onChange={handleUpload} />
            <div className="mt-5">
              <span className={clsx(buttonStyles, 'bg-white/6 text-sm', uploading && 'animate-pulse')}>
                {uploading ? 'Uploading...' : 'Select Video'}
              </span>
            </div>
          </label>

          <div className="mt-8">
            <div className="flex items-center justify-between">
              <p className="display-face text-[0.72rem] uppercase tracking-[0.32em] text-sky-100/70">Bundled Demo Clips</p>
              <span className="text-xs text-slate-400">{loadingSamples ? 'Loading...' : `${samples.length} available`}</span>
            </div>
            <div className="mt-4 grid gap-4 md:grid-cols-2">
              {samples.length ? (
                samples.map((sample) => (
                  <button
                    key={sample.id}
                    type="button"
                    onClick={() => setSelectedVideo(sample)}
                    className={clsx(
                      'panel-cut overflow-hidden p-3 text-left transition',
                      selectedVideo?.id === sample.id ? 'border-emerald-300/50 bg-emerald-300/10' : 'hover:border-sky-200/30',
                    )}
                  >
                    {sample.poster_url ? (
                      <img className="h-36 w-full object-cover" src={toAssetUrl(sample.poster_url)} alt={sample.name} />
                    ) : (
                      <div className="grid h-36 place-items-center bg-slate-900 text-sm text-slate-400">No poster available</div>
                    )}
                    <div className="mt-4 px-1 pb-1">
                      <p className="display-face text-[0.7rem] uppercase tracking-[0.32em] text-sky-100/70">Sample Clip</p>
                      <p className="mt-2 text-xl font-semibold">{sample.name}</p>
                      <p className="muted-copy mt-1 text-sm">{formatSeconds(sample.duration)} · {sample.width} Ã— {sample.height}</p>
                    </div>
                  </button>
                ))
              ) : (
                <div className="surface-outline rounded-none px-4 py-5 text-sm text-slate-400 md:col-span-2">
                  No bundled sample clips are present yet. Drop videos into <code>assets/sample-clips/</code> to surface them here.
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="space-y-6">
          <div className="panel-cut p-6">
            <p className="display-face text-[0.72rem] uppercase tracking-[0.32em] text-sky-100/70">Selected Source</p>
            {selectedVideo ? (
              <div className="mt-5 space-y-4">
                <p className="display-face text-3xl font-bold uppercase tracking-[0.14em]">{selectedVideo.name}</p>
                <div className="grid gap-3 sm:grid-cols-2">
                  <MetricCell label="Runtime" value={formatSeconds(selectedVideo.duration)} />
                  <MetricCell label="Source" value={selectedVideo.source_type === 'sample' ? 'Bundled sample' : 'Uploaded clip'} />
                </div>
                <video className="surface-outline h-64 w-full object-cover" src={toAssetUrl(selectedVideo.url)} controls preload="metadata" />
              </div>
            ) : (
              <p className="muted-copy mt-5 text-sm">Pick a sample or upload a file to lock this match to a video source.</p>
            )}
          </div>
          {error ? <div className="rounded-none border border-rose-300/20 bg-rose-300/10 px-4 py-3 text-sm text-rose-100">{error}</div> : null}
          <button
            className={clsx(buttonStyles, 'w-full bg-slate-50 text-slate-950 hover:bg-slate-100 disabled:cursor-wait disabled:opacity-70')}
            disabled={submitting}
            type="button"
            onClick={createLinkedMatch}
          >
            {submitting ? 'Preparing Console...' : 'Open Match Console'}
          </button>
        </div>
      </motion.section>
    </PageFrame>
  )
}

function MatchConsolePage({
  activeMatch,
  activeVideo,
  role,
  onIncident,
}: {
  activeMatch: MatchRecord | null
  activeVideo: VideoAsset | null
  role: Role
  onIncident: (incident: IncidentRecord) => void
}) {
  const navigate = useNavigate()
  const sourceVideoRef = useRef<HTMLVideoElement | null>(null)
  const reviewVideoRef = useRef<HTMLVideoElement | null>(null)
  const [reviewTime, setReviewTime] = useState(0)
  const [sourceInteracted, setSourceInteracted] = useState(false)
  const [processing, setProcessing] = useState<'offside' | 'goal' | null>(null)
  const [reviewClip, setReviewClip] = useState<ReviewClip | null>(null)
  const [reviewSelectionTime, setReviewSelectionTime] = useState(0)
  const [reviewPlayerReady, setReviewPlayerReady] = useState(false)
  const [reviewPlaying, setReviewPlaying] = useState(false)
  const [reviewExpanded, setReviewExpanded] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!activeVideo) return
    setSourceInteracted(false)
    setReviewTime(Math.max(0.04, activeVideo.duration - 0.04))
  }, [activeVideo])

  useEffect(() => {
    setReviewPlaying(false)
    setReviewPlayerReady(false)
    setReviewSelectionTime(0)
    setReviewExpanded(false)
  }, [reviewClip?.incident_id])

  if (role === 'Team Viewer') {
    return <Navigate to="/viewer" replace />
  }

  if (!activeMatch || !activeVideo) {
    return <Navigate to="/video" replace />
  }

  const primeSourceReviewTime = () => {
    if (!activeVideo) return
    setSourceInteracted(false)
    setReviewTime(Math.max(0.04, activeVideo.duration - 0.04))
  }

  const syncSourceTime = () => {
    if (!sourceVideoRef.current) return
    const currentTime = sourceVideoRef.current.currentTime
    if (!sourceInteracted && currentTime <= 0.12) {
      return
    }
    setSourceInteracted(true)
    setReviewTime(currentTime)
  }

  const syncReviewPlayback = () => {
    if (!reviewClip || !reviewVideoRef.current) return
    const player = reviewVideoRef.current
    const bounded = clamp(player.currentTime, 0, reviewClip.clip_duration)
    if (Math.abs(player.currentTime - bounded) > 0.02) {
      player.currentTime = bounded
    }
    if (player.currentTime >= reviewClip.clip_duration - 0.02) {
      player.pause()
      player.currentTime = reviewClip.clip_duration
      setReviewPlaying(false)
    }
    setReviewSelectionTime(clamp(player.currentTime, 0, reviewClip.clip_duration))
  }

  const initializeReviewWindow = () => {
    if (!reviewClip || !reviewVideoRef.current) return
    const player = reviewVideoRef.current
    player.currentTime = 0
    setReviewSelectionTime(0)
    setReviewPlayerReady(true)
    setReviewPlaying(!player.paused)
  }

  const seekReviewPlayer = (nextRelativeTime: number) => {
    if (!reviewClip) return
    const bounded = clamp(nextRelativeTime, 0, reviewClip.clip_duration)
    setReviewSelectionTime(bounded)
    if (reviewVideoRef.current) {
      reviewVideoRef.current.currentTime = bounded
    }
  }

  const resetReviewWindow = () => {
    if (!reviewClip || !reviewVideoRef.current) return
    const player = reviewVideoRef.current
    player.pause()
    player.currentTime = 0
    setReviewSelectionTime(0)
    setReviewPlaying(false)
    setReviewPlayerReady(true)
  }

  const stepReviewSelection = (delta: number) => {
    seekReviewPlayer(reviewRelativeTime + delta)
  }

  const toggleReviewPlayback = async () => {
    if (!reviewClip || !reviewVideoRef.current) return
    const player = reviewVideoRef.current
    if (!reviewPlayerReady) {
      initializeReviewWindow()
    }
    if (reviewPlaying) {
      player.pause()
      setReviewPlaying(false)
      return
    }
    if (player.currentTime < 0 || player.currentTime >= reviewClip.clip_duration - 0.02) {
      player.currentTime = 0
      setReviewSelectionTime(0)
    }
    try {
      await player.play()
      setReviewPlaying(true)
    } catch {
      setError('The review player could not start. Click inside the player lane and try play again.')
    }
  }

  const toggleReviewDeck = () => {
    setReviewExpanded((current) => !current)
  }

  const triggerReview = async (reviewType: 'offside' | 'goal') => {
    setProcessing(reviewType)
    setError('')
    try {
      const currentPlayerTime = sourceVideoRef.current?.currentTime ?? 0
      const triggerTime = sourceInteracted ? currentPlayerTime : reviewTime
      const clip = await createReviewClip({
        video_id: activeVideo.id,
        review_type: reviewType,
        review_timestamp: triggerTime,
      })
      setReviewClip(clip)
      setReviewSelectionTime(0)
      setReviewTime(triggerTime)
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : 'Review failed.')
    } finally {
      setProcessing(null)
    }
  }

  const submitFrameReview = async () => {
    if (!reviewClip || reviewClip.review_type !== 'offside') return
    setProcessing('offside')
    setError('')
    try {
      const incident = await reviewOffsideFrame({
        incident_id: reviewClip.incident_id,
        frame_timestamp: reviewClip.clip_start + reviewSelectionTime,
      })
      onIncident(incident)
      setReviewClip(null)
      startTransition(() => navigate(`/incidents/${incident.id}`))
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : 'Frame review failed.')
    } finally {
      setProcessing(null)
    }
  }

  const submitGoalReview = async () => {
    if (!reviewClip || reviewClip.review_type !== 'goal') return
    setProcessing('goal')
    setError('')
    try {
      const incident = await reviewGoal({ incident_id: reviewClip.incident_id })
      onIncident(incident)
      setReviewClip(null)
      startTransition(() => navigate(`/incidents/${incident.id}`))
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : 'Goal review failed.')
    } finally {
      setProcessing(null)
    }
  }

  const reviewSourceUrl = reviewClip?.clip_url ?? activeVideo.url
  const reviewWindowLabel = reviewClip
    ? `${formatSeconds(reviewSelectionTime)} / ${formatSeconds(reviewClip.clip_duration)}`
    : '00:00 / 00:00'
  const reviewRelativeTime = reviewSelectionTime

  return (
    <PageFrame key="console">
      <motion.section
        layout
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        className={clsx('grid gap-6', reviewExpanded ? 'xl:grid-cols-1' : 'xl:grid-cols-[1.24fr_0.96fr]')}
      >
        <motion.div layout className={clsx('panel-cut overflow-hidden p-4 md:p-6', reviewExpanded && 'xl:order-2')}>
          <div className="flex flex-col gap-3 border-b border-white/8 pb-4 md:flex-row md:items-end md:justify-between">
            <div>
              <p className="display-face text-[0.72rem] uppercase tracking-[0.32em] text-sky-100/70">Live Match Console</p>
              <h3 className="display-face mt-2 text-4xl font-bold uppercase tracking-[0.15em]">{activeMatch.home_team} vs {activeMatch.away_team}</h3>
            </div>
            <div className="grid gap-3 sm:grid-cols-3">
              <MetricCell label="Clip Source" value={activeVideo.source_type === 'sample' ? 'Sample' : 'Upload'} />
              <MetricCell label="Review Time" value={formatSeconds(reviewTime)} />
              <MetricCell label="Kickoff" value={activeMatch.kickoff.replace('T', ' ')} />
            </div>
          </div>

          <div className="mt-5 grid gap-5 xl:grid-cols-[1fr_0.36fr]">
            <div className="space-y-4">
              <video
                ref={sourceVideoRef}
                className={clsx('panel-cut w-full bg-slate-950 object-contain', reviewExpanded ? 'h-[340px] xl:h-[360px]' : 'h-[430px] xl:h-[510px]')}
                src={toAssetUrl(activeVideo.url)}
                controls
                preload="metadata"
                onLoadedMetadata={primeSourceReviewTime}
                onLoadedData={primeSourceReviewTime}
                onTimeUpdate={syncSourceTime}
                onSeeked={syncSourceTime}
                onPause={syncSourceTime}
                onPlay={() => setSourceInteracted(true)}
              />
              <div className="panel-cut pitch-grid px-4 py-4">
                <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
                  <div>
                    <p className="display-face text-[0.68rem] uppercase tracking-[0.3em] text-sky-100/70">Trigger the review from the live surface</p>
                    <p className="muted-copy mt-3 max-w-[56ch] text-sm leading-7 text-slate-300">Scrub to the review moment, then open Offside Check or Goal Check.</p>
                  </div>
                  <div className="surface-outline min-w-[180px] px-4 py-4">
                    <p className="display-face text-[0.62rem] uppercase tracking-[0.3em] text-slate-300/75">
                      {sourceInteracted ? 'Current source time' : 'Default review point'}
                    </p>
                    <p className="mt-2 text-3xl font-semibold">{formatSeconds(reviewTime)}</p>
                    <p className="mt-1 text-xs uppercase tracking-[0.22em] text-slate-400">
                      {sourceInteracted ? 'Using your selected moment' : 'Uses the end of the clip by default'}
                    </p>
                  </div>
                </div>
              </div>
            </div>

            <div className="flex flex-col gap-4">
              <button
                className={clsx(buttonStyles, 'review-action review-action--offside panel-cut w-full flex-1 flex-col items-start gap-4 bg-sky-300/8 p-5 text-left text-white')}
                type="button"
                disabled={processing !== null}
                onClick={() => triggerReview('offside')}
              >
                <span className="block text-[0.72rem] tracking-[0.32em] text-sky-100/70">10 Sec</span>
                <span className="mt-2 block text-3xl">Offside Check</span>
              </button>
              <button
                className={clsx(buttonStyles, 'review-action review-action--goal panel-cut w-full flex-1 flex-col items-start gap-4 bg-amber-300/10 p-5 text-left text-white')}
                type="button"
                disabled={processing !== null}
                onClick={() => triggerReview('goal')}
              >
                <span className="block text-[0.72rem] tracking-[0.32em] text-amber-100/70">5 Sec</span>
                <span className="mt-2 block text-3xl">Goal Check</span>
              </button>
              {error ? <div className="rounded-none border border-rose-300/20 bg-rose-300/10 px-4 py-3 text-sm text-rose-100">{error}</div> : null}
            </div>
          </div>
        </motion.div>

        <motion.div layout className={clsx('space-y-6', reviewExpanded && 'xl:order-1')}>
          {reviewClip ? (
            <motion.div layout className={clsx('panel-cut p-5', reviewExpanded && 'review-shell-expanded')}>
              <div className="flex flex-col gap-4 border-b border-white/8 pb-4 lg:flex-row lg:items-end lg:justify-between">
                <div>
                  <p className="display-face text-[0.72rem] uppercase tracking-[0.32em] text-sky-100/70">
                    {reviewClip.review_type === 'offside' ? 'Frame Lock Review' : 'Goal Sequence Review'}
                  </p>
                  <h4 className="display-face mt-3 text-3xl font-bold uppercase tracking-[0.14em]">
                    {reviewClip.review_type === 'offside' ? 'Choose the pass frame' : 'Inspect the goal window'}
                  </h4>
                  <p className="muted-copy mt-3 max-w-[64ch] text-sm leading-7 text-slate-300">
                    {reviewClip.review_type === 'offside'
                      ? 'This clip covers the 10 seconds before the review moment. Scrub to the touch of the final pass, then lock the frame.'
                      : 'This clip covers the 5 seconds before the review moment. Scrub the sequence, then run the goal review.'}
                  </p>
                </div>
                <div className="surface-outline px-4 py-3 text-right">
                  <p className="display-face text-[0.58rem] uppercase tracking-[0.26em] text-slate-300/75">Window</p>
                  <p className="mt-1 text-xl font-semibold">{formatSeconds(reviewClip.clip_duration)}</p>
                </div>
              </div>
              <div className={clsx('review-stage mt-5 p-3', reviewExpanded && 'review-stage--expanded')}>
                <div className="mb-3 flex items-center justify-between gap-3">
                  <div>
                    <p className="display-face text-[0.62rem] uppercase tracking-[0.28em] text-sky-100/70">Selected window</p>
                    <p className="mt-1 text-sm text-slate-300">{formatSeconds(reviewClip.clip_start)} to {formatSeconds(reviewClip.clip_end)}</p>
                  </div>
                  <div className="surface-outline px-4 py-3 text-right">
                    <p className="display-face text-[0.58rem] uppercase tracking-[0.26em] text-slate-300/75">Frame lock</p>
                    <p className="mt-1 text-xl font-semibold">{reviewWindowLabel}</p>
                  </div>
                </div>
                <video
                  key={reviewSourceUrl}
                  ref={reviewVideoRef}
                  className={clsx('w-full cursor-pointer bg-slate-950 object-contain', reviewExpanded ? 'h-[68vh] max-h-[840px]' : 'h-[26rem] xl:h-[34rem]')}
                  src={toAssetUrl(reviewSourceUrl)}
                  controls
                  preload="auto"
                  muted
                  crossOrigin="anonymous"
                  playsInline
                  onLoadedMetadata={initializeReviewWindow}
                  onLoadedData={initializeReviewWindow}
                  onCanPlay={() => setReviewPlayerReady(true)}
                  onDurationChange={() => setReviewPlayerReady(true)}
                  onTimeUpdate={syncReviewPlayback}
                  onSeeked={syncReviewPlayback}
                  onPlay={() => setReviewPlaying(true)}
                  onPause={() => setReviewPlaying(false)}
                />
              </div>
              <div className="mt-4 grid gap-3 sm:grid-cols-[1fr_auto_auto] xl:grid-cols-[1fr_auto_auto]">
                <button
                  className={clsx(buttonStyles, 'w-full bg-white/6 text-slate-100 hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-60')}
                  type="button"
                  disabled={!reviewPlayerReady}
                  onClick={toggleReviewPlayback}
                >
                  {reviewPlaying ? 'Pause Window' : 'Play Window'}
                </button>
                <button
                  className={clsx(subtleButtonStyles, 'disabled:cursor-not-allowed disabled:opacity-60')}
                  type="button"
                  disabled={!reviewPlayerReady}
                  onClick={resetReviewWindow}
                >
                  Reset To Start
                </button>
                <button
                  className={clsx(subtleButtonStyles, 'disabled:cursor-not-allowed disabled:opacity-60')}
                  type="button"
                  disabled={!reviewPlayerReady}
                  onClick={toggleReviewDeck}
                >
                  {reviewExpanded ? 'Collapse' : 'Expand'}
                </button>
              </div>
              <div className="mt-5">
                <div className="flex items-center justify-between text-sm text-slate-300">
                  <span>{reviewClip.review_type === 'offside' ? 'Locked frame time' : 'Window cursor'}</span>
                  <span>{reviewWindowLabel}</span>
                </div>
                <input
                  className="timeline-input mt-4 w-full"
                  type="range"
                  min={0}
                  max={reviewClip.clip_duration}
                  step={0.04}
                  value={reviewRelativeTime}
                  onChange={(event) => seekReviewPlayer(Number(event.target.value))}
                />
                <div className="mt-3 flex items-center justify-between text-xs uppercase tracking-[0.24em] text-slate-400">
                  <span>{formatSeconds(reviewClip.clip_start)} window start</span>
                  <span>{formatSeconds(reviewClip.clip_end)} window end</span>
                </div>
                <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-4">
                  <button className={clsx(subtleButtonStyles, 'review-step')} type="button" onClick={() => stepReviewSelection(-1)}>
                    -1.0s
                  </button>
                  <button className={clsx(subtleButtonStyles, 'review-step')} type="button" onClick={() => stepReviewSelection(-0.04)}>
                    Prev Frame
                  </button>
                  <button className={clsx(subtleButtonStyles, 'review-step')} type="button" onClick={() => stepReviewSelection(0.04)}>
                    Next Frame
                  </button>
                  <button className={clsx(subtleButtonStyles, 'review-step')} type="button" onClick={() => stepReviewSelection(1)}>
                    +1.0s
                  </button>
                </div>
                <p className="muted-copy mt-4 text-sm">
                  {reviewClip.review_type === 'offside'
                    ? 'Scrub or nudge until the final pass contact is exact, then lock the frame.'
                    : 'Scrub or play through the window, then launch the goal review.'}
                </p>
              </div>
              <button
                className={clsx(buttonStyles, 'mt-5 w-full bg-slate-50 text-slate-950 hover:bg-slate-100 disabled:cursor-wait disabled:opacity-70')}
                type="button"
                disabled={processing !== null}
                onClick={reviewClip.review_type === 'offside' ? submitFrameReview : submitGoalReview}
              >
                {reviewClip.review_type === 'offside'
                  ? processing === 'offside'
                    ? 'Reviewing Frame...'
                    : 'Review This Frame'
                  : processing === 'goal'
                    ? 'Scanning Goal Sequence...'
                    : 'Run Goal Review'}
              </button>
            </motion.div>
          ) : (
            <motion.div layout className="panel-cut p-5">
              <p className="display-face text-[0.72rem] uppercase tracking-[0.32em] text-sky-100/70">Ready State</p>
              <h4 className="display-face mt-3 text-3xl font-bold uppercase tracking-[0.14em]">Awaiting review trigger</h4>
              <p className="muted-copy mt-4 text-sm leading-7">
                Trigger either review from the left. The right-side lane will open with the correct review window so you can scrub the pass touch or preview the goal sequence before sending it through the engine.
              </p>
            </motion.div>
          )}
          <button className={clsx(subtleButtonStyles, 'w-full')} type="button" onClick={() => navigate('/incidents')}>
            Open Incident Log
          </button>
        </motion.div>
      </motion.section>
    </PageFrame>
  )
}

function IncidentLogPage({
  role,
  incidentCache,
  setIncidentCache,
}: {
  role: Role
  incidentCache: Record<string, IncidentRecord>
  setIncidentCache: (value: Record<string, IncidentRecord>) => void
}) {
  const navigate = useNavigate()
  const [incidents, setIncidents] = useState<IncidentRecord[]>(Object.values(incidentCache))
  const [loading, setLoading] = useState(false)
  const [deletingId, setDeletingId] = useState<string | null>(null)
  const [confirmDeleteIncident, setConfirmDeleteIncident] = useState<IncidentRecord | null>(null)
  const [query, setQuery] = useState('')
  const deferredQuery = useDeferredValue(query)

  useEffect(() => {
    setLoading(true)
    getIncidents()
      .then((records) => {
        setIncidents(records)
        setIncidentCache(Object.fromEntries(records.map((record) => [record.id, record])))
      })
      .finally(() => setLoading(false))
  }, [setIncidentCache])

  const filteredIncidents = useMemo(() => {
    if (!deferredQuery.trim()) return incidents
    const normalized = deferredQuery.toLowerCase()
    return incidents.filter((incident) => {
      const verdict = incident.verdict ?? ''
      return [incident.id, incident.review_type, verdict, incident.rationale ?? ''].join(' ').toLowerCase().includes(normalized)
    })
  }, [deferredQuery, incidents])

  const removeIncident = async (incidentId: string) => {
    if (role === 'Team Viewer') return
    setDeletingId(incidentId)
    try {
      await deleteIncident(incidentId)
      const nextIncidents = incidents.filter((incident) => incident.id !== incidentId)
      setIncidents(nextIncidents)
      setIncidentCache(Object.fromEntries(nextIncidents.map((incident) => [incident.id, incident])))
      setConfirmDeleteIncident(null)
    } finally {
      setDeletingId(null)
    }
  }

  return (
    <PageFrame key="incident-log">
      <motion.section initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
        <div className="panel-cut p-5 md:p-6">
          <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
            <div>
              <p className="display-face text-[0.72rem] uppercase tracking-[0.32em] text-sky-100/70">Incident Archive</p>
              <h3 className="surface-heading mt-2 text-[2.85rem] text-slate-50">Every trimmed check in one queue</h3>
            </div>
            <input
              className="surface-outline w-full max-w-sm px-4 py-3 text-sm outline-none placeholder:text-slate-500"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search by verdict, type, or rationale..."
            />
          </div>
        </div>

        <div className="grid gap-4 md:grid-cols-3">
          <MetricCell label="Total incidents" value={String(incidents.length).padStart(2, '0')} />
          <MetricCell label="High confidence" value={String(incidents.filter((item) => (item.confidence ?? 0) >= 0.7).length).padStart(2, '0')} />
          <MetricCell label="Needs human review" value={String(incidents.filter((item) => item.verdict === 'Human Review').length).padStart(2, '0')} />
        </div>

        <div className="grid gap-4">
          {loading ? (
            <div className="panel-cut p-6 text-sm text-slate-300">Loading incidents...</div>
          ) : filteredIncidents.length ? (
            filteredIncidents.map((incident) => (
              <div
                key={incident.id}
                className="panel-cut grid gap-4 p-5 text-left transition hover:border-sky-200/30 md:grid-cols-[0.9fr_1.2fr_0.8fr_0.8fr_auto]"
              >
                <button className="contents text-left" onClick={() => navigate(`/incidents/${incident.id}`)} type="button">
                  <div>
                    <p className="display-face text-[0.68rem] uppercase tracking-[0.3em] text-sky-100/70">{reviewTypeLabel(incident.review_type)}</p>
                    <p className="mt-2 text-2xl font-semibold">{formatSeconds(incident.review_timestamp)}</p>
                  </div>
                  <div>
                    <p className="text-sm text-slate-300">{incident.rationale ?? 'Awaiting analysis details.'}</p>
                  </div>
                  <MetricBadge label="Verdict" value={incident.verdict ?? 'Pending'} />
                  <MetricBadge label="Confidence" value={incident.confidence ? `${Math.round(incident.confidence * 100)}%` : 'N/A'} />
                </button>
                <div className="flex items-center justify-end gap-2">
                  <button className={clsx(subtleButtonStyles, 'px-3 py-2')} onClick={() => navigate(`/incidents/${incident.id}`)} type="button">
                    Open
                  </button>
                  {role !== 'Team Viewer' ? (
                    <button
                      className={clsx(subtleButtonStyles, 'border-rose-300/18 bg-rose-300/8 px-3 py-2 text-rose-100 hover:border-rose-300/34 hover:bg-rose-300/14')}
                      disabled={deletingId === incident.id}
                      onClick={() => setConfirmDeleteIncident(incident)}
                      type="button"
                    >
                      {deletingId === incident.id ? 'Deleting' : 'Delete'}
                    </button>
                  ) : null}
                </div>
              </div>
            ))
          ) : (
            <div className="panel-cut p-6 text-sm text-slate-300">No incidents match the current search.</div>
          )}
        </div>
      </motion.section>
      <ConfirmDialog
        open={Boolean(confirmDeleteIncident)}
        title="Delete incident"
        body={
          confirmDeleteIncident
            ? `Remove ${reviewTypeLabel(confirmDeleteIncident.review_type)} at ${formatSeconds(confirmDeleteIncident.review_timestamp)} and its generated media from the local archive?`
            : ''
        }
        confirmLabel="Delete incident"
        busyLabel="Deleting..."
        busy={Boolean(deletingId)}
        onCancel={() => {
          if (!deletingId) setConfirmDeleteIncident(null)
        }}
        onConfirm={() => {
          if (confirmDeleteIncident) void removeIncident(confirmDeleteIncident.id)
        }}
      />
    </PageFrame>
  )
}

function IncidentDetailPage({
  role,
  incidentCache,
  setIncidentCache,
}: {
  role: Role
  incidentCache: Record<string, IncidentRecord>
  setIncidentCache: (value: Record<string, IncidentRecord>) => void
}) {
  const params = useParams()
  const incidentId = params.incidentId ?? ''
  const [incident, setIncident] = useState<IncidentRecord | null>(incidentCache[incidentId] ?? null)
  const [note, setNote] = useState(incidentCache[incidentId]?.note ?? '')
  const [saving, setSaving] = useState(false)
  const [expandedMedia, setExpandedMedia] = useState<ExpandedMedia | null>(null)
  const [correctionOpen, setCorrectionOpen] = useState(false)
  const [correctionTarget, setCorrectionTarget] = useState<CorrectionTarget>('attacker')
  const [correctionDirection, setCorrectionDirection] = useState<AttackDirection>('right')
  const [selectedAttackerId, setSelectedAttackerId] = useState<string | null>(null)
  const [selectedDefenderId, setSelectedDefenderId] = useState<string | null>(null)
  const [manualAttackerPoint, setManualAttackerPoint] = useState<[number, number] | null>(null)
  const [manualDefenderPoint, setManualDefenderPoint] = useState<[number, number] | null>(null)
  const [applyingCorrection, setApplyingCorrection] = useState(false)

  useEffect(() => {
    if (!incidentId) return
    getIncident(incidentId).then((record) => {
      setIncident(record)
      setNote(record.note ?? '')
      setIncidentCache({ ...incidentCache, [record.id]: record })
    })
  }, [incidentId, setIncidentCache])

  useEffect(() => {
    if (!incident) return
    setSelectedAttackerId(getSelectionDefault(incident, 'attacker'))
    setSelectedDefenderId(getSelectionDefault(incident, 'defender'))
    setManualAttackerPoint(null)
    setManualDefenderPoint(null)
    setCorrectionDirection(getIncidentAttackDirection(incident))
    setCorrectionTarget('attacker')
  }, [incident])

  if (!incident) {
    return (
      <PageFrame key="incident-detail-empty">
        <div className="panel-cut p-6 text-sm text-slate-300">Loading incident detail...</div>
      </PageFrame>
    )
  }

  const saveNote = async () => {
    setSaving(true)
    try {
      const updated = await saveIncidentNote(incident.id, note)
      setIncident(updated)
      setIncidentCache({ ...incidentCache, [updated.id]: updated })
    } finally {
      setSaving(false)
    }
  }

  const canManualCorrect = role !== 'Team Viewer' && incident.review_type === 'offside'
  const diagnosticSignals = summarizeDiagnostics(incident)
  const assetVersion = incident.updated_at

  const applyCorrection = async () => {
    const hasAttacker = !!selectedAttackerId || !!manualAttackerPoint
    const hasDefender = !!selectedDefenderId || !!manualDefenderPoint
    if (!incident || !hasAttacker || !hasDefender) return
    if (selectedAttackerId && selectedDefenderId && selectedAttackerId === selectedDefenderId) return
    setApplyingCorrection(true)
    try {
      const updated = await applyOffsideCorrection(incident.id, {
        attacker_id: selectedAttackerId,
        defender_id: selectedDefenderId,
        attacker_point: manualAttackerPoint,
        defender_point: manualDefenderPoint,
        attack_direction: correctionDirection,
      })
      setIncident(updated)
      setIncidentCache({ ...incidentCache, [updated.id]: updated })
      setCorrectionOpen(false)
    } finally {
      setApplyingCorrection(false)
    }
  }

  return (
    <PageFrame key={incident.id}>
      <motion.section initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="grid gap-6 xl:grid-cols-[1.15fr_0.85fr]">
        <div className="space-y-6">
          <div className="panel-cut p-6">
            <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
              <div>
                <p className="display-face text-[0.72rem] uppercase tracking-[0.32em] text-sky-100/70">Incident Detail</p>
                <h3 className="display-face mt-2 text-5xl font-bold uppercase tracking-[0.14em]">{incident.verdict ?? 'Pending'}</h3>
              </div>
              <div className="grid gap-3 sm:grid-cols-2 xl:min-w-[340px]">
                <MetricBadge label="Type" value={reviewTypeLabel(incident.review_type)} />
                <MetricBadge label="Confidence" value={incident.confidence ? `${Math.round(incident.confidence * 100)}%` : 'N/A'} />
                {canManualCorrect ? (
                  <button
                    className={clsx(subtleButtonStyles, 'sm:col-span-2 px-4 py-3 text-xs uppercase tracking-[0.24em]')}
                    type="button"
                    onClick={() => setCorrectionOpen(true)}
                  >
                    Adjust Attacker / Defender
                  </button>
                ) : null}
              </div>
            </div>

            <video
              className="panel-cut mt-6 h-72 w-full bg-slate-950 object-contain"
              src={toAssetUrl(incident.source_video_url ?? incident.clip_url)}
              controls
              preload="metadata"
            />
            <div className="mt-4 flex items-center justify-between text-xs uppercase tracking-[0.24em] text-slate-400">
              <span>{formatSeconds(incident.clip_start)} review window start</span>
              <span>{formatSeconds(incident.clip_end)} review window end</span>
            </div>

            <div className="mt-6 grid gap-5 lg:grid-cols-2">
              <MediaCard
                title="Annotated Snapshot"
                src={incident.snapshot_url}
                alt="Incident snapshot"
                version={assetVersion}
                onExpand={(src) => setExpandedMedia({ src, title: 'Annotated Snapshot', alt: 'Incident snapshot', version: assetVersion })}
              />
              <MediaCard
                title="Position Diagram"
                src={incident.diagram_url}
                alt="Position diagram"
                emptyCopy="Not generated for this review type."
                version={assetVersion}
                onExpand={(src) => setExpandedMedia({ src, title: 'Position Diagram', alt: 'Position diagram', version: assetVersion })}
              />
            </div>
          </div>
        </div>

        <div className="space-y-6">
          <div className="panel-cut p-6">
            <p className="display-face text-[0.72rem] uppercase tracking-[0.32em] text-sky-100/70">AI Rationale</p>
            <p className="mt-4 text-base leading-7 text-slate-100">{incident.rationale ?? 'Awaiting rationale.'}</p>
          </div>
          <div className="panel-cut p-6">
            <div className="flex items-center justify-between gap-4">
              <div>
                <p className="display-face text-[0.72rem] uppercase tracking-[0.32em] text-sky-100/70">Signals</p>
              </div>
              <span className="review-badge">{reviewTypeLabel(incident.review_type)}</span>
            </div>
            <div className="mt-5 grid gap-3 sm:grid-cols-2">
              {diagnosticSignals.map((item) => (
                <MetricCell key={item.label} label={item.label} value={item.value} />
              ))}
            </div>
          </div>
          <div className="panel-cut p-6">
            <p className="display-face text-[0.72rem] uppercase tracking-[0.32em] text-sky-100/70">Referee Note</p>
            <textarea
              value={note}
              onChange={(event) => setNote(event.target.value)}
              disabled={role === 'Team Viewer'}
              maxLength={300}
              className="surface-outline mt-4 min-h-36 w-full px-4 py-3 text-sm outline-none placeholder:text-slate-500 disabled:cursor-not-allowed disabled:opacity-70"
              placeholder="Add a concise note about why this review was triggered..."
            />
            {role !== 'Team Viewer' ? (
              <button
                className={clsx(buttonStyles, 'mt-4 w-full bg-slate-50 text-slate-950 hover:bg-slate-100 disabled:cursor-wait disabled:opacity-70')}
                type="button"
                disabled={saving}
                onClick={saveNote}
              >
                {saving ? 'Saving Note...' : 'Save Referee Note'}
              </button>
            ) : (
              <p className="muted-copy mt-4 text-sm">Team Viewer access is read-only. Notes are visible but not editable.</p>
            )}
          </div>
        </div>
      </motion.section>
      {incident ? (
        <OffsideCorrectionModal
          open={correctionOpen}
          incident={incident}
          selectedAttackerId={selectedAttackerId}
          selectedDefenderId={selectedDefenderId}
          manualAttackerPoint={manualAttackerPoint}
          manualDefenderPoint={manualDefenderPoint}
          correctionDirection={correctionDirection}
          correctionTarget={correctionTarget}
          applying={applyingCorrection}
          onClose={() => setCorrectionOpen(false)}
          onPickAttacker={(candidateId) => {
            setSelectedAttackerId(candidateId)
            setManualAttackerPoint(null)
            if (candidateId === selectedDefenderId) {
              setSelectedDefenderId(null)
            }
            setCorrectionTarget('defender')
          }}
          onPickDefender={(candidateId) => {
            setSelectedDefenderId(candidateId)
            setManualDefenderPoint(null)
            if (candidateId === selectedAttackerId) {
              setSelectedAttackerId(null)
            }
            setCorrectionTarget('attacker')
          }}
          onMarkManualPoint={(point) => {
            if (correctionTarget === 'attacker') {
              setSelectedAttackerId(null)
              setManualAttackerPoint(point)
              setCorrectionTarget('defender')
              return
            }
            setSelectedDefenderId(null)
            setManualDefenderPoint(point)
            setCorrectionTarget('attacker')
          }}
          onSetTarget={setCorrectionTarget}
          onSetDirection={setCorrectionDirection}
          onApply={applyCorrection}
        />
      ) : null}
      <MediaLightbox media={expandedMedia} onClose={() => setExpandedMedia(null)} />
    </PageFrame>
  )
}

function TeamViewerPage({
  incidentCache,
  setIncidentCache,
}: {
  incidentCache: Record<string, IncidentRecord>
  setIncidentCache: (value: Record<string, IncidentRecord>) => void
}) {
  const navigate = useNavigate()
  const [incidents, setIncidents] = useState<IncidentRecord[]>(Object.values(incidentCache))

  useEffect(() => {
    getIncidents().then((records) => {
      setIncidents(records)
      setIncidentCache(Object.fromEntries(records.map((record) => [record.id, record])))
    })
  }, [setIncidentCache])

  return (
    <PageFrame key="viewer">
      <motion.section initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
        <div className="panel-cut p-6">
          <p className="display-face text-[0.72rem] uppercase tracking-[0.32em] text-sky-100/70">Team Viewer Surface</p>
          <h3 className="surface-heading mt-3 text-[2.85rem] text-slate-50">Approved clips and verdicts</h3>
          <p className="muted-copy mt-4 max-w-[56ch] text-base leading-7">
            This surface is read-only. Team viewers can open incident clips, see the result assets, and read the final referee note after the incident is logged.
          </p>
        </div>
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {incidents.map((incident) => (
            <button
              key={incident.id}
              type="button"
              onClick={() => navigate(`/incidents/${incident.id}`)}
              className="panel-cut overflow-hidden text-left transition hover:border-sky-200/30"
            >
              {incident.snapshot_url ? (
                <img className="h-48 w-full object-cover" src={toAssetUrl(incident.snapshot_url)} alt={incident.id} />
              ) : (
                <div className="grid h-48 place-items-center bg-slate-950 text-sm text-slate-400">Awaiting snapshot</div>
              )}
              <div className="space-y-3 p-5">
                <div className="flex items-center justify-between">
                  <span className="display-face text-[0.68rem] uppercase tracking-[0.3em] text-sky-100/70">{reviewTypeLabel(incident.review_type)}</span>
                  <span className="text-xs text-slate-400">{formatSeconds(incident.review_timestamp)}</span>
                </div>
                <p className="display-face text-3xl font-bold uppercase tracking-[0.14em]">{incident.verdict ?? 'Pending'}</p>
                <p className="text-sm text-slate-300">{incident.note || 'No referee note recorded yet.'}</p>
              </div>
            </button>
          ))}
        </div>
      </motion.section>
    </PageFrame>
  )
}

function AppNavItem({ to, label }: { to: string; label: string }) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        clsx(
          'display-face block rounded-none border px-4 py-3 text-sm uppercase tracking-[0.26em] transition',
          isActive ? 'border-sky-200/40 bg-sky-300/12 text-white' : 'border-white/8 text-slate-300 hover:border-white/18 hover:bg-white/6',
        )
      }
    >
      {label}
    </NavLink>
  )
}

function PageFrame({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      transition={{ duration: 0.28, ease: 'easeOut' }}
      className={clsx('space-y-6', className)}
    >
      {children}
    </motion.div>
  )
}

function MetricCell({ label, value }: { label: string; value: string }) {
  return (
    <div className="panel-cut min-w-0 px-4 py-3">
      <p className="display-face truncate text-[0.64rem] uppercase tracking-[0.3em] text-sky-100/70">{label}</p>
      <p className="mt-2 truncate text-base font-semibold">{value}</p>
    </div>
  )
}

function MetricBadge({ label, value }: { label: string; value: string }) {
  return (
    <div className="surface-outline px-4 py-3">
      <p className="display-face text-[0.62rem] uppercase tracking-[0.3em] text-slate-300/75">{label}</p>
      <p className="mt-2 text-lg font-semibold">{value}</p>
    </div>
  )
}

function MetricNarrative({ title, body }: { title: string; body: string }) {
  return (
    <div className="surface-outline p-4">
      <p className="display-face text-[0.68rem] uppercase tracking-[0.28em] text-sky-100/70">{title}</p>
      <p className="mt-3 text-sm leading-6 text-slate-300">{body}</p>
    </div>
  )
}

function FootballMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 48 48" fill="none" className={className} aria-hidden="true">
      <circle cx="24" cy="24" r="20" className="fill-slate-100/95 stroke-sky-100/40" strokeWidth="2" />
      <path d="M24 13l6 4.4-2.3 7.1h-7.4L18 17.4 24 13z" className="fill-slate-900/90" />
      <path d="M16.7 18.6l-4.8 4.7 1.4 7 6.7 1.7m11-13.4l4.9 4.7-1.5 7-6.6 1.7m-9.3.2l3.3 5.7h5.5l3.3-5.7m-18-7.4l2.9-8m17.6 0l2.8 8" className="stroke-slate-900/82" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx="24" cy="24" r="20" className="stroke-emerald-300/35" strokeWidth="1.2" />
    </svg>
  )
}

function TutorialStep({ index, title, body }: { index: string; title: string; body: string }) {
  return (
    <div className="panel-cut bg-white/[0.02] p-4">
      <p className="display-face text-[0.66rem] uppercase tracking-[0.28em] text-emerald-200/78">{index}</p>
      <h3 className="surface-heading mt-3 text-[1.55rem] font-semibold text-slate-50">{title}</h3>
      <p className="muted-copy mt-3 text-sm leading-6">{body}</p>
    </div>
  )
}

function FootballHeroGraphic() {
  return (
    <div className="w-full max-w-[30rem] space-y-4">
      <div className="login-hero__field panel-cut px-5 py-5">
        <motion.div
          className="login-hero__goal-line"
          animate={{ opacity: [0.26, 0.92, 0.26], scaleY: [0.82, 1.02, 0.82] }}
          transition={{ duration: 3.4, repeat: Infinity, ease: 'easeInOut' }}
        />
        <div className="login-hero__goal-frame" />
        <motion.div
          className="login-hero__ball login-hero__ball--goal"
          animate={{ x: [0, 84, 150, 226], y: [0, -10, -22, -40], rotate: [0, 100, 220, 360] }}
          transition={{ duration: 5.4, repeat: Infinity, ease: 'easeInOut' }}
        />
        <motion.div
          className="login-hero__ball-shadow"
          animate={{ x: [0, 84, 150, 226], scaleX: [1, 0.9, 0.84, 0.72], opacity: [0.24, 0.18, 0.14, 0.08] }}
          transition={{ duration: 5.4, repeat: Infinity, ease: 'easeInOut' }}
        />
        <motion.div
          className="login-hero__shot-trail"
          animate={{ scaleX: [0.12, 0.64, 1, 0.3], opacity: [0.16, 0.78, 0.92, 0.18] }}
          transition={{ duration: 5.4, repeat: Infinity, ease: 'easeInOut' }}
        />
        <div className="absolute bottom-5 left-5 right-5 flex items-end justify-between text-xs uppercase tracking-[0.22em] text-slate-300/78">
          <div>
            <p className="display-face text-[0.62rem] tracking-[0.28em] text-sky-100/70">Active Review</p>
            <p className="mt-2 text-slate-100">Ball path · goal line · verdict</p>
          </div>
          <div className="text-right">
            <p className="display-face text-[0.62rem] tracking-[0.28em] text-sky-100/70">Workflow</p>
            <p className="mt-2 text-slate-100">Clip · lock · decide</p>
          </div>
        </div>
      </div>
      <div className="login-hero__ticker overflow-hidden px-4 py-3">
        <motion.div
          className="login-hero__track display-face text-[0.74rem] uppercase tracking-[0.3em] text-slate-200/88"
          animate={{ x: ['0%', '-50%'] }}
          transition={{ duration: 15, repeat: Infinity, ease: 'linear' }}
        >
          <span>OFFSIDE CHECK</span>
          <span>GOAL CHECK</span>
          <span>FRAME LOCK</span>
          <span>INCIDENT LOG</span>
          <span>OFFSIDE CHECK</span>
          <span>GOAL CHECK</span>
          <span>FRAME LOCK</span>
          <span>INCIDENT LOG</span>
        </motion.div>
      </div>
    </div>
  )
}

function Field({
  label,
  value,
  onChange,
  placeholder,
  type = 'text',
}: {
  label: string
  value: string
  onChange: (value: string) => void
  placeholder: string
  type?: string
}) {
  return (
    <label className="block">
      <span className="display-face text-[0.72rem] uppercase tracking-[0.28em] text-slate-300/80">{label}</span>
      <input
        className="surface-outline mt-3 w-full px-4 py-3 outline-none placeholder:text-slate-500"
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder={placeholder}
        type={type}
      />
    </label>
  )
}

function MediaCard({
  title,
  src,
  alt,
  emptyCopy,
  version,
  onExpand,
}: {
  title: string
  src?: string | null
  alt: string
  emptyCopy?: string
  version?: string | null
  onExpand?: (src: string) => void
}) {
  return (
    <div className="panel-cut p-4">
      <div className="flex items-center justify-between gap-3">
        <p className="display-face text-[0.68rem] uppercase tracking-[0.28em] text-sky-100/70">{title}</p>
        {src ? (
          <button className={clsx(subtleButtonStyles, 'px-3 py-1.5 text-[0.66rem] tracking-[0.18em]')} type="button" onClick={() => onExpand?.(src)}>
            Open
          </button>
        ) : null}
      </div>
      {src ? (
        <div className="review-stage mt-4 p-3">
          <img className="h-72 w-full object-contain" src={toAssetUrl(src, version)} alt={alt} />
        </div>
      ) : (
        <div className="mt-4 grid h-64 place-items-center bg-slate-950 text-sm text-slate-400">{emptyCopy ?? 'Not available yet.'}</div>
      )}
    </div>
  )
}

function MediaLightbox({ media, onClose }: { media: ExpandedMedia | null; onClose: () => void }) {
  useEffect(() => {
    if (!media) return
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [media, onClose])

  return (
    <AnimatePresence>
      {media ? (
        <motion.div
          className="fixed inset-0 z-[70] flex items-center justify-center bg-slate-950/88 px-4 py-6 backdrop-blur-md"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={onClose}
        >
          <motion.div
            className="panel-cut review-lightbox w-full max-w-[1500px] p-4 md:p-6"
            initial={{ opacity: 0, y: 26, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 18, scale: 0.985 }}
            transition={{ duration: 0.22, ease: 'easeOut' }}
            onClick={(event) => event.stopPropagation()}
          >
            <div className="flex items-center justify-between gap-4 border-b border-white/8 pb-4">
              <div>
                <p className="display-face text-[0.66rem] uppercase tracking-[0.28em] text-sky-100/70">Full View</p>
                <h4 className="display-face mt-2 text-3xl font-bold uppercase tracking-[0.12em]">{media.title}</h4>
              </div>
              <button className={clsx(subtleButtonStyles, 'px-4 py-2')} type="button" onClick={onClose}>
                Close
              </button>
            </div>
            <div className="review-stage mt-5 p-4">
              <img className="max-h-[78vh] w-full object-contain" src={toAssetUrl(media.src, media.version)} alt={media.alt} />
            </div>
          </motion.div>
        </motion.div>
      ) : null}
    </AnimatePresence>
  )
}

function ConfirmDialog({
  open,
  title,
  body,
  confirmLabel,
  busyLabel,
  busy,
  onCancel,
  onConfirm,
}: {
  open: boolean
  title: string
  body: string
  confirmLabel: string
  busyLabel: string
  busy?: boolean
  onCancel: () => void
  onConfirm: () => void
}) {
  useEffect(() => {
    if (!open) return
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !busy) onCancel()
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [busy, onCancel, open])

  return (
    <AnimatePresence>
      {open ? (
        <motion.div
          className="fixed inset-0 z-[95] flex items-center justify-center bg-slate-950/82 px-4 py-6 backdrop-blur-md"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={() => {
            if (!busy) onCancel()
          }}
        >
          <motion.div
            className="panel-cut w-full max-w-[520px] p-6"
            initial={{ opacity: 0, y: 18, scale: 0.985 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 10, scale: 0.985 }}
            transition={{ duration: 0.2, ease: 'easeOut' }}
            onClick={(event) => event.stopPropagation()}
          >
            <p className="display-face text-[0.7rem] uppercase tracking-[0.28em] text-rose-200/76">Confirm action</p>
            <h4 className="surface-heading mt-3 text-[2rem] font-semibold text-slate-50">{title}</h4>
            <p className="muted-copy mt-4 text-sm leading-7">{body}</p>
            <div className="mt-7 flex flex-wrap justify-end gap-3">
              <button className={clsx(subtleButtonStyles, 'px-4 py-2')} type="button" disabled={busy} onClick={onCancel}>
                Cancel
              </button>
              <button
                className={clsx(buttonStyles, 'bg-rose-200 px-5 py-2.5 text-slate-950 hover:bg-rose-100 disabled:cursor-wait disabled:opacity-70')}
                type="button"
                disabled={busy}
                onClick={onConfirm}
              >
                {busy ? busyLabel : confirmLabel}
              </button>
            </div>
          </motion.div>
        </motion.div>
      ) : null}
    </AnimatePresence>
  )
}

function OffsideCorrectionModal({
  open,
  incident,
  selectedAttackerId,
  selectedDefenderId,
  manualAttackerPoint,
  manualDefenderPoint,
  correctionDirection,
  correctionTarget,
  applying,
  onClose,
  onPickAttacker,
  onPickDefender,
  onMarkManualPoint,
  onSetTarget,
  onSetDirection,
  onApply,
}: {
  open: boolean
  incident: IncidentRecord
  selectedAttackerId: string | null
  selectedDefenderId: string | null
  manualAttackerPoint: [number, number] | null
  manualDefenderPoint: [number, number] | null
  correctionDirection: AttackDirection
  correctionTarget: CorrectionTarget
  applying: boolean
  onClose: () => void
  onPickAttacker: (candidateId: string) => void
  onPickDefender: (candidateId: string) => void
  onMarkManualPoint: (point: [number, number]) => void
  onSetTarget: (target: CorrectionTarget) => void
  onSetDirection: (direction: AttackDirection) => void
  onApply: () => void
}) {
  const [focusCandidateId, setFocusCandidateId] = useState<string | null>(null)
  const [showFringeCandidates, setShowFringeCandidates] = useState(false)
  const [zoomScale, setZoomScale] = useState(2.4)
  const [manualPlacementMode, setManualPlacementMode] = useState(false)

  useEffect(() => {
    if (!open) return
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [open, onClose])

  useEffect(() => {
    if (!open) return
    setFocusCandidateId(
      selectedAttackerId ??
        selectedDefenderId ??
        incident.player_candidates.find((candidate) => isLikelyInPlay(candidate))?.id ??
        incident.player_candidates[0]?.id ??
        null,
    )
    setShowFringeCandidates(false)
    setZoomScale(2.4)
    setManualPlacementMode(false)
  }, [open, incident, selectedAttackerId, selectedDefenderId])

  const frameWidth = readDiagnosticNumber(incident, 'frame_width') ?? 1
  const frameHeight = readDiagnosticNumber(incident, 'frame_height') ?? 1
  const hasAttackerSelection = Boolean(selectedAttackerId || manualAttackerPoint)
  const hasDefenderSelection = Boolean(selectedDefenderId || manualDefenderPoint)
  const duplicateDetectedSelection =
    !!selectedAttackerId &&
    !!selectedDefenderId &&
    !manualAttackerPoint &&
    !manualDefenderPoint &&
    selectedAttackerId === selectedDefenderId
  const canApply = hasAttackerSelection && hasDefenderSelection && !duplicateDetectedSelection && !applying
  const frameImageUrl = toAssetUrl(incident.frame_source_url ?? incident.snapshot_url, incident.updated_at)

  const rankedCandidates = useMemo(
    () => [...incident.player_candidates].sort((left, right) => sortManualCandidates(left, right, selectedAttackerId, selectedDefenderId)),
    [incident.player_candidates, selectedAttackerId, selectedDefenderId],
  )

  const visibleCandidates = useMemo(() => {
    if (showFringeCandidates) return rankedCandidates
    const primary = rankedCandidates.filter((candidate) => isLikelyInPlay(candidate) || candidate.id === selectedAttackerId || candidate.id === selectedDefenderId)
    if (primary.length >= 6) return primary
    const fallback = rankedCandidates.filter(
      (candidate) => (candidate.on_pitch ?? true) || (candidate.pitch_score ?? 0) >= 0.32 || candidate.id === selectedAttackerId || candidate.id === selectedDefenderId,
    )
    return fallback.length ? fallback : rankedCandidates
  }, [rankedCandidates, selectedAttackerId, selectedDefenderId, showFringeCandidates])

  const hiddenFringeCount = Math.max(0, rankedCandidates.length - visibleCandidates.length)
  const focusCandidate =
    incident.player_candidates.find((candidate) => candidate.id === focusCandidateId) ??
    incident.player_candidates.find((candidate) => candidate.id === selectedAttackerId) ??
    incident.player_candidates.find((candidate) => candidate.id === selectedDefenderId) ??
    visibleCandidates[0] ??
    null

  const handleCandidatePick = (candidateId: string) => {
    setFocusCandidateId(candidateId)
    setManualPlacementMode(false)
    if (correctionTarget === 'attacker') {
      onPickAttacker(candidateId)
      return
    }
    onPickDefender(candidateId)
  }

  const handleManualPoint = (point: [number, number]) => {
    setManualPlacementMode(true)
    onMarkManualPoint(point)
  }

  const focusPoint = correctionTarget === 'attacker' ? manualAttackerPoint : manualDefenderPoint

  return (
    <AnimatePresence>
      {open ? (
        <motion.div
          className="fixed inset-0 z-[80] flex items-center justify-center bg-slate-950/90 px-3 py-4 backdrop-blur-md md:px-6"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={onClose}
        >
          <motion.div
            className="panel-cut correction-shell flex h-[min(92vh,980px)] w-full max-w-[1680px] flex-col overflow-hidden"
            initial={{ opacity: 0, y: 28, scale: 0.985 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 18, scale: 0.985 }}
            transition={{ duration: 0.24, ease: 'easeOut' }}
            onClick={(event) => event.stopPropagation()}
          >
            <div className="flex flex-col gap-5 border-b border-white/8 px-5 py-5 lg:flex-row lg:items-end lg:justify-between lg:px-7">
              <div className="space-y-3">
                <p className="display-face text-[0.66rem] uppercase tracking-[0.32em] text-sky-100/70">Manual Offside Correction</p>
                <div>
                  <h4 className="display-face text-4xl font-bold uppercase tracking-[0.14em] lg:text-5xl">Refine The Key Players</h4>
                  <p className="muted-copy mt-3 max-w-[72ch] text-sm leading-6">
                    Click the attacker and defending reference directly on the frame. Use the attack direction toggle if the play is moving the other way, then apply the corrected review.
                  </p>
                </div>
              </div>
              <div className="flex flex-wrap items-center gap-3">
                <button
                  className={clsx(subtleButtonStyles, 'px-5 py-3 text-xs uppercase tracking-[0.24em]')}
                  type="button"
                  onClick={onClose}
                >
                  Close
                </button>
                <button
                  className={clsx(
                    buttonStyles,
                    'px-6 py-3 text-sm',
                    canApply ? 'bg-slate-50 text-slate-950 hover:bg-slate-100' : 'cursor-not-allowed bg-white/8 text-slate-400 hover:bg-white/8',
                  )}
                  type="button"
                  disabled={!canApply}
                  onClick={onApply}
                >
                  {applying ? 'Applying Correction...' : 'Apply Corrected Review'}
                </button>
              </div>
            </div>

            <div className="grid min-h-0 flex-1 gap-4 px-4 py-4 lg:grid-cols-[1.28fr_0.72fr] lg:px-5">
              <div className="review-stage review-stage--expanded flex min-h-0 flex-col p-4">
                <div className="mb-4 flex flex-col gap-3 xl:flex-row xl:items-center xl:justify-between">
                  <div className="flex flex-wrap items-center gap-3">
                    <button
                      type="button"
                      onClick={() => onSetTarget('attacker')}
                      className={clsx(
                        'correction-toggle',
                        correctionTarget === 'attacker' ? 'correction-toggle--active-attacker' : 'correction-toggle--idle',
                      )}
                    >
                      1 / Pick attacker
                    </button>
                    <button
                      type="button"
                      onClick={() => onSetTarget('defender')}
                      className={clsx(
                        'correction-toggle',
                        correctionTarget === 'defender' ? 'correction-toggle--active-defender' : 'correction-toggle--idle',
                      )}
                    >
                      2 / Pick defender
                    </button>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => setManualPlacementMode((value) => !value)}
                      className={clsx('correction-direction', manualPlacementMode && 'correction-direction--active')}
                    >
                      {manualPlacementMode ? 'Manual Marking On' : `Mark Missing ${correctionTarget === 'attacker' ? 'Attacker' : 'Defender'}`}
                    </button>
                    <button
                      type="button"
                      onClick={() => onSetDirection('left')}
                      className={clsx('correction-direction', correctionDirection === 'left' && 'correction-direction--active')}
                    >
                      Attack Left
                    </button>
                    <button
                      type="button"
                      onClick={() => onSetDirection('right')}
                      className={clsx('correction-direction', correctionDirection === 'right' && 'correction-direction--active')}
                    >
                      Attack Right
                    </button>
                  </div>
                </div>

                <div className="mb-3 flex flex-wrap items-center gap-3 text-xs uppercase tracking-[0.24em] text-slate-300/80">
                  <span className="review-badge">{correctionTarget === 'attacker' ? 'Attacker selection armed' : 'Defender selection armed'}</span>
                  <span className="review-badge">
                    {manualPlacementMode ? 'Manual placement active - click frame or zoom pane' : 'Click frame, zoom pane, or chip'}
                  </span>
                  <span className="review-badge">{visibleCandidates.length} active candidates</span>
                  {hiddenFringeCount > 0 ? (
                    <button
                      type="button"
                      onClick={() => setShowFringeCandidates((value) => !value)}
                      className={clsx(subtleButtonStyles, 'px-3 py-2 text-[10px] uppercase tracking-[0.22em]')}
                    >
                      {showFringeCandidates ? 'Hide fringe detections' : `Show fringe detections (${hiddenFringeCount})`}
                    </button>
                  ) : null}
                </div>

                {frameImageUrl ? (
                  <CandidateSelectionStage
                    imageUrl={frameImageUrl}
                    frameWidth={frameWidth}
                    frameHeight={frameHeight}
                    candidates={visibleCandidates}
                    selectedAttackerId={selectedAttackerId}
                    selectedDefenderId={selectedDefenderId}
                    manualAttackerPoint={manualAttackerPoint}
                    manualDefenderPoint={manualDefenderPoint}
                    focusCandidateId={focusCandidate?.id ?? null}
                    manualPlacementMode={manualPlacementMode}
                    onFocus={setFocusCandidateId}
                    onPick={handleCandidatePick}
                    onMarkManualPoint={handleManualPoint}
                  />
                ) : (
                  <div className="grid flex-1 place-items-center border border-white/8 bg-slate-950 text-sm text-slate-400">
                    Snapshot unavailable for manual correction.
                  </div>
                )}

                <div className="mt-4 flex flex-wrap gap-3">
                  {visibleCandidates.map((candidate) => {
                    const isAttacker = candidate.id === selectedAttackerId
                    const isDefender = candidate.id === selectedDefenderId
                    return (
                      <button
                        key={candidate.id}
                        type="button"
                        onClick={() => handleCandidatePick(candidate.id)}
                        onMouseEnter={() => setFocusCandidateId(candidate.id)}
                        className={clsx(
                          'candidate-chip',
                          isAttacker && 'candidate-chip--attacker',
                          isDefender && 'candidate-chip--defender',
                          !isAttacker && !isDefender && 'candidate-chip--idle',
                        )}
                      >
                        <span className="display-face text-sm uppercase tracking-[0.18em]">{candidateShortLabel(candidate)}</span>
                        <span className="text-[11px] uppercase tracking-[0.18em] text-slate-300/78">
                          {candidate.team ? `${candidate.team} · ` : ''}
                          {Math.round(candidate.confidence * 100)}%
                        </span>
                      </button>
                    )
                  })}
                </div>
              </div>

              <div className="panel-cut flex min-h-0 flex-col p-5">
                <div className="space-y-5">
                  <div className="surface-outline p-4">
                    <div className="flex items-center justify-between gap-3">
                      <div>
                        <p className="display-face text-[0.66rem] uppercase tracking-[0.28em] text-slate-300/78">Focus Zoom</p>
                        <p className="mt-2 text-sm text-slate-200">
                          {manualSummary(correctionTarget === 'attacker' ? manualAttackerPoint : manualDefenderPoint, correctionTarget) ??
                            (focusCandidate ? candidateSummary(incident.player_candidates, focusCandidate.id) : 'Hover or click a player to inspect them up close.')}
                        </p>
                      </div>
                      <div className="flex items-center gap-2">
                        {[1.8, 2.4, 3.2].map((scale) => (
                          <button
                            key={scale}
                            type="button"
                            onClick={() => setZoomScale(scale)}
                            className={clsx('correction-direction px-3 py-2 text-[10px]', Math.abs(zoomScale - scale) < 0.01 && 'correction-direction--active')}
                          >
                            {scale.toFixed(1)}x
                          </button>
                        ))}
                      </div>
                    </div>
                    <CandidateFocusCard
                      imageUrl={frameImageUrl}
                      candidate={focusCandidate}
                      manualPoint={focusPoint}
                      manualPlacementMode={manualPlacementMode}
                      frameWidth={frameWidth}
                      frameHeight={frameHeight}
                      zoomScale={zoomScale}
                      onMarkManualPoint={handleManualPoint}
                    />
                  </div>

                  <div>
                    <p className="display-face text-[0.68rem] uppercase tracking-[0.28em] text-sky-100/70">Selection Summary</p>
                    <div className="mt-4 grid gap-3">
                      <SelectionCard
                        tone="attacker"
                        label="Attacker"
                        value={candidateSummary(incident.player_candidates, selectedAttackerId) ?? manualSummary(manualAttackerPoint, 'attacker') ?? 'Not selected yet'}
                        active={correctionTarget === 'attacker'}
                        onClick={() => onSetTarget('attacker')}
                      />
                      <SelectionCard
                        tone="defender"
                        label="Defender"
                        value={candidateSummary(incident.player_candidates, selectedDefenderId) ?? manualSummary(manualDefenderPoint, 'defender') ?? 'Not selected yet'}
                        active={correctionTarget === 'defender'}
                        onClick={() => onSetTarget('defender')}
                      />
                    </div>
                  </div>

                  <div className="surface-outline p-4">
                    <p className="display-face text-[0.66rem] uppercase tracking-[0.28em] text-slate-300/78">Attack Direction</p>
                    <p className="mt-3 text-sm leading-6 text-slate-200">
                      The offside line is evaluated toward the active goal. Flip the direction if the attacking team is moving the other way.
                    </p>
                    <div className="mt-4 grid gap-2 sm:grid-cols-2">
                      <button
                        type="button"
                        onClick={() => onSetDirection('left')}
                        className={clsx('correction-direction w-full justify-center', correctionDirection === 'left' && 'correction-direction--active')}
                      >
                        Left
                      </button>
                      <button
                        type="button"
                        onClick={() => onSetDirection('right')}
                        className={clsx('correction-direction w-full justify-center', correctionDirection === 'right' && 'correction-direction--active')}
                      >
                        Right
                      </button>
                    </div>
                  </div>

                  <div className="surface-outline p-4">
                    <p className="display-face text-[0.66rem] uppercase tracking-[0.28em] text-slate-300/78">How To Use It</p>
                    <ol className="mt-4 space-y-3 text-sm leading-6 text-slate-200">
                      <li>Hover a candidate to inspect them in the zoom pane.</li>
                      <li>Click the true attacker, then switch to defender mode and click the defender.</li>
                      <li>If a player is missing entirely, turn on manual marking and click the planted foot on the frame or in the zoom pane.</li>
                    </ol>
                  </div>

                  <div className="surface-outline p-4">
                    <p className="display-face text-[0.66rem] uppercase tracking-[0.28em] text-slate-300/78">Why sideline people still appear</p>
                    <p className="mt-3 text-sm leading-6 text-slate-200">
                      The detector sees generic people, not football players. Bright touchlines and crowded goalmouths can weaken the pitch mask, so fringe detections are hidden by default and only shown if you choose to reveal them.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </motion.div>
      ) : null}
    </AnimatePresence>
  )
}

function CandidateFocusCard({
  imageUrl,
  candidate,
  manualPoint,
  manualPlacementMode,
  frameWidth,
  frameHeight,
  zoomScale,
  onMarkManualPoint,
}: {
  imageUrl: string
  candidate: PlayerCandidate | null
  manualPoint: [number, number] | null
  manualPlacementMode: boolean
  frameWidth: number
  frameHeight: number
  zoomScale: number
  onMarkManualPoint: (point: [number, number]) => void
}) {
  if (!imageUrl) {
    return <div className="correction-focus mt-4 grid place-items-center text-sm text-slate-400">No player in focus yet.</div>
  }

  const focusX =
    manualPoint?.[0] ??
    (candidate ? (candidate.bbox[0] + candidate.bbox[2]) / 2 : frameWidth / 2)
  const focusY =
    manualPoint?.[1] ??
    (candidate ? (candidate.bbox[1] + candidate.bbox[3]) / 2 : frameHeight / 2)
  const centerX = (focusX / frameWidth) * 100
  const centerY = (focusY / frameHeight) * 100

  const handleZoomPick = (event: ReactMouseEvent<HTMLDivElement>) => {
    if (!manualPlacementMode) return
    const rect = event.currentTarget.getBoundingClientRect()
    const localX = clamp((event.clientX - rect.left) / Math.max(1, rect.width), 0, 1)
    const localY = clamp((event.clientY - rect.top) / Math.max(1, rect.height), 0, 1)
    const visibleWidth = frameWidth / zoomScale
    const visibleHeight = frameHeight / zoomScale
    const nextX = clamp(focusX + (localX - 0.5) * visibleWidth, 0, frameWidth)
    const nextY = clamp(focusY + (localY - 0.5) * visibleHeight, 0, frameHeight)
    onMarkManualPoint([Math.round(nextX), Math.round(nextY)])
  }

  return (
    <div className={clsx('correction-focus mt-4', manualPlacementMode && 'correction-focus--manual')} onClick={handleZoomPick}>
      <div
        className="correction-focus__image"
        style={{
          backgroundImage: `url("${imageUrl}")`,
          backgroundPosition: `${centerX}% ${centerY}%`,
          backgroundSize: `${zoomScale * 100}%`,
        }}
      />
      <div className="correction-focus__crosshair" />
    </div>
  )
}

function CandidateSelectionStage({
  imageUrl,
  frameWidth,
  frameHeight,
  candidates,
  selectedAttackerId,
  selectedDefenderId,
  manualAttackerPoint,
  manualDefenderPoint,
  focusCandidateId,
  manualPlacementMode,
  onFocus,
  onPick,
  onMarkManualPoint,
}: {
  imageUrl: string
  frameWidth: number
  frameHeight: number
  candidates: PlayerCandidate[]
  selectedAttackerId: string | null
  selectedDefenderId: string | null
  manualAttackerPoint: [number, number] | null
  manualDefenderPoint: [number, number] | null
  focusCandidateId: string | null
  manualPlacementMode: boolean
  onFocus: (candidateId: string) => void
  onPick: (candidateId: string) => void
  onMarkManualPoint: (point: [number, number]) => void
}) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const imageRef = useRef<HTMLImageElement | null>(null)
  const [viewport, setViewport] = useState({ left: 0, top: 0, width: 0, height: 0 })

  useEffect(() => {
    const element = containerRef.current
    if (!element) return

    const updateViewport = () => {
      const containerWidth = element.clientWidth
      const containerHeight = element.clientHeight
      if (!containerWidth || !containerHeight || !frameWidth || !frameHeight) return

      const frameRatio = frameWidth / frameHeight
      const containerRatio = containerWidth / containerHeight

      if (containerRatio > frameRatio) {
        const height = containerHeight
        const width = height * frameRatio
        setViewport({ left: (containerWidth - width) / 2, top: 0, width, height })
      } else {
        const width = containerWidth
        const height = width / frameRatio
        setViewport({ left: 0, top: (containerHeight - height) / 2, width, height })
      }
    }

    updateViewport()
    const observer = new ResizeObserver(updateViewport)
    observer.observe(element)
    window.addEventListener('resize', updateViewport)
    return () => {
      observer.disconnect()
      window.removeEventListener('resize', updateViewport)
    }
  }, [frameWidth, frameHeight])

  const handleBackgroundPick = (event: ReactMouseEvent<HTMLDivElement>) => {
    if (!manualPlacementMode) return
    const element = imageRef.current?.getBoundingClientRect() ?? event.currentTarget.getBoundingClientRect()
    if (viewport.width <= 0 || viewport.height <= 0) return
    const localX = event.clientX - element.left
    const localY = event.clientY - element.top
    const imageWidth = Math.max(1, element.width)
    const imageHeight = Math.max(1, element.height)
    if (localX < 0 || localY < 0 || localX > imageWidth || localY > imageHeight) return
    const frameX = (localX / imageWidth) * frameWidth
    const frameY = (localY / imageHeight) * frameHeight
    onMarkManualPoint([Math.round(frameX), Math.round(frameY)])
  }

  return (
    <div
      ref={containerRef}
      className="relative flex-1 overflow-hidden border border-white/8 bg-slate-950"
      style={{ aspectRatio: `${frameWidth} / ${frameHeight}` }}
    >
      <div
        className={clsx('absolute', manualPlacementMode && 'cursor-crosshair')}
        style={{
          left: `${viewport.left}px`,
          top: `${viewport.top}px`,
          width: `${viewport.width}px`,
          height: `${viewport.height}px`,
        }}
        onClick={handleBackgroundPick}
      >
        <img ref={imageRef} className="h-full w-full object-fill" src={imageUrl} alt="Manual correction source frame" />
        <div className="absolute inset-0">
          {candidates.map((candidate) => {
            const [x1, y1, x2, y2] = candidate.bbox
            const left = `${(x1 / frameWidth) * 100}%`
            const top = `${(y1 / frameHeight) * 100}%`
            const width = `${((x2 - x1) / frameWidth) * 100}%`
            const height = `${((y2 - y1) / frameHeight) * 100}%`
            const isAttacker = candidate.id === selectedAttackerId
            const isDefender = candidate.id === selectedDefenderId
            const isFocused = candidate.id === focusCandidateId
            const showTag = isAttacker || isDefender || isFocused
            const activeTone = isAttacker
              ? 'correction-box--attacker'
              : isDefender
                ? 'correction-box--defender'
                : 'correction-box--idle'

            return (
              <button
                key={candidate.id}
                type="button"
                className={clsx(
                  'correction-box',
                  activeTone,
                  isFocused && 'correction-box--focused',
                  manualPlacementMode && 'correction-box--muted',
                  manualPlacementMode && 'pointer-events-none',
                  showTag ? 'correction-box--tagged' : 'correction-box--untagged',
                )}
                style={{ left, top, width, height }}
                onClick={(event) => {
                  if (manualPlacementMode) return
                  event.stopPropagation()
                  onPick(candidate.id)
                }}
                onMouseEnter={() => {
                  if (!manualPlacementMode) onFocus(candidate.id)
                }}
                title={candidateTitle(candidate)}
              >
                <span className="correction-box__tag">{candidateShortLabel(candidate)}</span>
                <span className="correction-box__feet" style={{ left: `${(((candidate.feet_point[0] - x1) / Math.max(1, x2 - x1)) * 100).toFixed(2)}%` }} />
              </button>
            )
          })}
          {manualAttackerPoint ? (
            <ManualPointMarker point={manualAttackerPoint} frameWidth={frameWidth} frameHeight={frameHeight} tone="attacker" />
          ) : null}
          {manualDefenderPoint ? (
            <ManualPointMarker point={manualDefenderPoint} frameWidth={frameWidth} frameHeight={frameHeight} tone="defender" />
          ) : null}
        </div>
      </div>
    </div>
  )
}

function ManualPointMarker({
  point,
  frameWidth,
  frameHeight,
  tone,
}: {
  point: [number, number]
  frameWidth: number
  frameHeight: number
  tone: CorrectionTarget
}) {
  return (
    <div
      className={clsx('manual-point-marker', tone === 'attacker' ? 'manual-point-marker--attacker' : 'manual-point-marker--defender')}
      style={{
        left: `${(point[0] / frameWidth) * 100}%`,
        top: `${(point[1] / frameHeight) * 100}%`,
      }}
    >
      <span className="manual-point-marker__tag">{tone === 'attacker' ? 'Manual Attacker' : 'Manual Defender'}</span>
      <span className="manual-point-marker__dot" />
    </div>
  )
}

function SelectionCard({
  tone,
  label,
  value,
  active,
  onClick,
}: {
  tone: CorrectionTarget
  label: string
  value: string
  active: boolean
  onClick: () => void
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={clsx(
        'surface-outline flex items-center justify-between gap-4 p-4 text-left transition',
        active && (tone === 'attacker' ? 'border-amber-300/45 bg-amber-200/8' : 'border-emerald-300/42 bg-emerald-200/8'),
      )}
    >
      <div>
        <p className="display-face text-[0.64rem] uppercase tracking-[0.28em] text-slate-300/78">{label}</p>
        <p className="mt-2 text-base font-semibold text-slate-100">{value}</p>
      </div>
      <span className="review-badge">{active ? 'Armed' : 'Pick'}</span>
    </button>
  )
}

function headlineForPath(pathname: string) {
  if (pathname.startsWith('/video')) return 'Load Video'
  if (pathname.startsWith('/console')) return 'Match Console'
  if (pathname.startsWith('/incidents/')) return 'Incident Detail'
  if (pathname.startsWith('/incidents')) return 'Incident Log'
  if (pathname.startsWith('/viewer')) return 'Team Viewer'
  return 'Create Match'
}

function formatSeconds(value: number) {
  if (!Number.isFinite(value)) return '00:00'
  const totalSeconds = Math.max(0, Math.floor(value))
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`
}

function reviewTypeLabel(reviewType: ReviewType) {
  return reviewType === 'offside' ? 'Offside Check' : 'Goal Check'
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

function getSelectionDefault(incident: IncidentRecord, target: CorrectionTarget) {
  const selected = target === 'attacker' ? incident.selected_attacker_id : incident.selected_defender_id
  if (selected) return selected

  const suggestionLabel = target === 'attacker' ? 'attacker' : 'defender'
  const suggestion = incident.suggestions.find((item) => item.label === suggestionLabel)
  if (suggestion) return suggestion.id

  const fallbackIndex = target === 'attacker' ? 0 : 1
  return incident.player_candidates[fallbackIndex]?.id ?? incident.player_candidates[0]?.id ?? null
}

function sortManualCandidates(
  left: PlayerCandidate,
  right: PlayerCandidate,
  selectedAttackerId: string | null,
  selectedDefenderId: string | null,
) {
  const scoreCandidate = (candidate: PlayerCandidate) => {
    let score = 0
    if (candidate.id === selectedAttackerId || candidate.id === selectedDefenderId) score += 100
    if (candidate.on_pitch !== false) score += 20
    score += (candidate.pitch_score ?? 0) * 50
    score += candidate.confidence * 20
    return score
  }
  return scoreCandidate(right) - scoreCandidate(left)
}

function isLikelyInPlay(candidate: PlayerCandidate) {
  if (candidate.on_pitch === false) return false
  const pitchScore = candidate.pitch_score ?? 0.5
  if (pitchScore >= 0.5) return true
  return pitchScore >= 0.35 && candidate.confidence >= 0.5
}

function getIncidentAttackDirection(incident: IncidentRecord): AttackDirection {
  if (incident.attack_direction === 'left' || incident.attack_direction === 'right') {
    return incident.attack_direction
  }
  const diagnosticValue = incident.diagnostics.attack_direction
  return diagnosticValue === 'left' ? 'left' : 'right'
}

function readDiagnosticNumber(incident: IncidentRecord, key: string) {
  const value = incident.diagnostics[key]
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function candidateShortLabel(candidate: PlayerCandidate) {
  const chunks = candidate.id.split('_')
  const suffix = chunks[chunks.length - 1]
  return `P-${suffix.toUpperCase()}`
}

function candidateTitle(candidate: PlayerCandidate) {
  return `${candidateShortLabel(candidate)}${candidate.team ? ` · ${candidate.team}` : ''}`
}

function candidateSummary(candidates: PlayerCandidate[], candidateId: string | null) {
  if (!candidateId) return 'Not selected yet'
  const candidate = candidates.find((item) => item.id === candidateId)
  if (!candidate) return 'Not selected yet'
  return `${candidateShortLabel(candidate)}${candidate.team ? ` · ${candidate.team}` : ''}`
}

function manualSummary(point: [number, number] | null, target: CorrectionTarget) {
  if (!point) return null
  return target === 'attacker' ? 'Manual attacker mark' : 'Manual defender mark'
}

function summarizeDiagnostics(incident: IncidentRecord) {
  const diagnostics = incident.diagnostics
  const entries: Array<{ label: string; value: string }> = []

  const add = (label: string, value: string | null | undefined) => {
    if (!value) return
    entries.push({ label, value })
  }

  if (incident.review_type === 'offside') {
    add('Selection', formatDiagnosticWord(diagnostics.selection_mode))
    add('Attack', formatAttackDirection(diagnostics.attack_direction))
    add('Frame Mode', formatDiagnosticWord(diagnostics.context_mode))
    add('Ball Lock', formatBooleanSignal(diagnostics.ball_detected || diagnostics.passer_locked))
    add('Line Quality', formatPercentSignal(diagnostics.line_strength))
    add('Team Split', formatPercentSignal(diagnostics.cluster_score))
    add('Players', formatCountSignal(diagnostics.pitch_player_count ?? diagnostics.player_count))
  } else {
    add('Sampled Frames', formatCountSignal(diagnostics.sample_count))
    add('Ball Lock', formatBooleanSignal(diagnostics.ball_detected))
    add('Goal Line', formatBooleanSignal(diagnostics.line_found ?? diagnostics.goal_line_visible))
    add('Line Quality', formatPercentSignal(diagnostics.line_strength))
    add('Clear Frames', formatCountSignal(diagnostics.crossings))
    add('Boundary', formatDiagnosticWord(diagnostics.boundary_state))
    add('Gap', formatPixelSignal(diagnostics.clearance_px))
  }

  return entries.slice(0, 6)
}

function formatDiagnosticWord(value: unknown) {
  if (typeof value !== 'string' || !value) return null
  return value
    .split(/[_\s-]+/)
    .filter(Boolean)
    .map((chunk) => chunk.charAt(0).toUpperCase() + chunk.slice(1))
    .join(' ')
}

function formatAttackDirection(value: unknown) {
  if (value === 'left') return 'Left'
  if (value === 'right') return 'Right'
  return null
}

function formatBooleanSignal(value: unknown) {
  if (typeof value !== 'boolean') return null
  return value ? 'Confirmed' : 'Not Locked'
}

function formatPercentSignal(value: unknown) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return null
  return `${Math.round(value * 100)}%`
}

function formatCountSignal(value: unknown) {
  if (typeof value === 'number' && Number.isFinite(value)) return `${Math.round(value)}`
  return null
}

function formatPixelSignal(value: unknown) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return null
  return `${value >= 0 ? '+' : ''}${value.toFixed(1)} px`
}

export default App

