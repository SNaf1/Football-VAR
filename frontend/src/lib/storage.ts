import { useEffect, useState } from 'react'

export function readStoredState<T>(key: string, fallback: T): T {
  try {
    const value = window.localStorage.getItem(key)
    return value ? (JSON.parse(value) as T) : fallback
  } catch {
    return fallback
  }
}

export function usePersistentState<T>(key: string, fallback: T) {
  const [value, setValue] = useState<T>(() => readStoredState(key, fallback))

  useEffect(() => {
    try {
      window.localStorage.setItem(key, JSON.stringify(value))
    } catch {
      // Ignore storage failures so the UI still runs in private mode or strict environments.
    }
  }, [key, value])

  return [value, setValue] as const
}
