import type {
  AttackDirection,
  DemoSession,
  IncidentRecord,
  MatchRecord,
  ReviewClip,
  Role,
  ReviewType,
  VideoAsset,
} from '../types'

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000').replace(/\/$/, '')

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, options)
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: 'Request failed.' }))
    throw new Error(payload.detail ?? 'Request failed.')
  }
  return response.json() as Promise<T>
}

export function toAssetUrl(path?: string | null, version?: string | number | null) {
  if (!path) return ''
  const base = /^https?:\/\//.test(path) ? encodeURI(path) : `${API_BASE_URL}${encodeURI(path)}`
  if (version === undefined || version === null || version === '') return base
  const separator = base.includes('?') ? '&' : '?'
  return `${base}${separator}v=${encodeURIComponent(String(version))}`
}

export function healthcheck() {
  return request<{ status: string; environment: string }>('/health')
}

export function demoLogin(role: Role, displayName: string, email?: string) {
  return request<DemoSession>('/auth/demo-login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ role, display_name: displayName, email }),
  })
}

export function getSampleClips() {
  return request<VideoAsset[]>('/sample-clips')
}

export async function uploadVideo(file: File) {
  const formData = new FormData()
  formData.append('file', file)
  return request<VideoAsset>('/uploads/video', {
    method: 'POST',
    body: formData,
  })
}

export function createMatch(payload: {
  title: string
  home_team: string
  away_team: string
  kickoff: string
  video_id: string
}) {
  return request<MatchRecord>('/matches', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
}

export function createReviewClip(payload: {
  video_id: string
  review_type: ReviewType
  review_timestamp: number
}) {
  return request<ReviewClip>('/reviews/clip', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
}

export function reviewOffsideFrame(payload: { incident_id: string; frame_timestamp: number }) {
  return request<IncidentRecord>('/reviews/offside/frame', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
}

export function reviewGoal(payload: { incident_id: string }) {
  return request<IncidentRecord>('/reviews/goal/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
}

export function getIncidents() {
  return request<IncidentRecord[]>('/incidents')
}

export function getIncident(incidentId: string) {
  return request<IncidentRecord>(`/incidents/${incidentId}`)
}

export function saveIncidentNote(incidentId: string, note: string) {
  return request<IncidentRecord>(`/incidents/${incidentId}/note`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ note }),
  })
}

export function deleteIncident(incidentId: string) {
  return request<{ id: string; status: string }>(`/incidents/${incidentId}`, {
    method: 'DELETE',
  })
}

export function applyOffsideCorrection(
  incidentId: string,
  payload: {
    attacker_id?: string | null
    defender_id?: string | null
    attacker_point?: [number, number] | null
    defender_point?: [number, number] | null
    attack_direction: AttackDirection
  },
) {
  return request<IncidentRecord>(`/incidents/${incidentId}/offside-correction`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
}
