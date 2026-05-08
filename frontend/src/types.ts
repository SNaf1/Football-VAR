export type Role = 'Match Official' | 'Team Viewer'
export type ReviewType = 'offside' | 'goal'
export type ReviewStatus = 'pending_frame_lock' | 'processing' | 'reviewed'
export type Verdict = 'Offside' | 'Onside' | 'Goal' | 'No Goal' | 'Human Review'
export type AttackDirection = 'left' | 'right'

export interface DemoSession {
  session_id: string
  role: Role
  display_name: string
  email?: string | null
  created_at: string
}

export interface DraftMatch {
  title: string
  homeTeam: string
  awayTeam: string
  kickoff: string
}

export interface VideoAsset {
  id: string
  name: string
  source_type: 'upload' | 'sample'
  duration: number
  fps: number
  width: number
  height: number
  url: string
  poster_url?: string | null
  created_at: string
}

export interface MatchRecord {
  id: string
  title: string
  home_team: string
  away_team: string
  kickoff: string
  video_id: string
  status: string
  created_at: string
}

export interface ReviewClip {
  incident_id: string
  match_id?: string | null
  review_type: ReviewType
  clip_start: number
  clip_end: number
  clip_duration: number
  clip_url: string
  source_video_url?: string | null
  status: ReviewStatus
}

export interface DetectionSuggestion {
  id: string
  label: string
  team?: string | null
  confidence: number
  bbox: number[]
}

export interface PlayerCandidate {
  id: string
  team?: string | null
  confidence: number
  bbox: number[]
  feet_point: number[]
  pitch_score?: number | null
  on_pitch?: boolean | null
}

export interface IncidentRecord {
  id: string
  match_id?: string | null
  video_id: string
  review_type: ReviewType
  review_timestamp: number
  clip_start: number
  clip_end: number
  clip_url: string
  source_video_url?: string | null
  status: ReviewStatus
  verdict?: Verdict | null
  confidence?: number | null
  frame_timestamp?: number | null
  frame_source_url?: string | null
  snapshot_url?: string | null
  diagram_url?: string | null
  note?: string | null
  rationale?: string | null
  diagnostics: Record<string, unknown>
  suggestions: DetectionSuggestion[]
  player_candidates: PlayerCandidate[]
  selected_attacker_id?: string | null
  selected_defender_id?: string | null
  attack_direction?: AttackDirection | null
  created_at: string
  updated_at: string
}
