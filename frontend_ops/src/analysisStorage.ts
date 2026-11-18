const DATABASE_NAME = 'nav-fund-analytics'
const DATABASE_VERSION = 1
const STORE_NAME = 'analysis'
const ANALYSIS_KEY = 'latest-analysis'

export type SavedAnalysis<T> = {
  results: T
  selectedFundIndex: number
  savedAt: string
}

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    if (typeof window === 'undefined' || !('indexedDB' in window)) {
      reject(new Error('IndexedDB is not supported by this browser.'))
      return
    }

    const request = window.indexedDB.open(
      DATABASE_NAME,
      DATABASE_VERSION,
    )

    request.onupgradeneeded = () => {
      const database = request.result

      if (!database.objectStoreNames.contains(STORE_NAME)) {
        database.createObjectStore(STORE_NAME)
      }
    }

    request.onsuccess = () => {
      const database = request.result

      database.onversionchange = () => {
        database.close()
      }

      resolve(database)
    }

    request.onerror = () => {
      reject(
        request.error ?? new Error('Could not open analysis storage.'),
      )
    }

    request.onblocked = () => {
      reject(
        new Error('Analysis storage is blocked by another browser tab.'),
      )
    }
  })
}

export async function saveAnalysis<T>(
  results: T,
  selectedFundIndex: number,
): Promise<void> {
  const database = await openDatabase()

  return new Promise((resolve, reject) => {
    const transaction = database.transaction(STORE_NAME, 'readwrite')
    const store = transaction.objectStore(STORE_NAME)

    const savedAnalysis: SavedAnalysis<T> = {
      results,
      selectedFundIndex,
      savedAt: new Date().toISOString(),
    }

    store.put(savedAnalysis, ANALYSIS_KEY)

    transaction.oncomplete = () => {
      database.close()
      resolve()
    }

    transaction.onerror = () => {
      database.close()
      reject(
        transaction.error ?? new Error('Could not save analysis results.'),
      )
    }

    transaction.onabort = () => {
      database.close()
      reject(
        transaction.error ?? new Error('Saving analysis results was aborted.'),
      )
    }
  })
}

export async function loadAnalysis<T>(): Promise<SavedAnalysis<T> | null> {
  const database = await openDatabase()

  return new Promise((resolve, reject) => {
    const transaction = database.transaction(STORE_NAME, 'readonly')
    const store = transaction.objectStore(STORE_NAME)
    const request = store.get(ANALYSIS_KEY)

    let savedAnalysis: SavedAnalysis<T> | null = null

    request.onsuccess = () => {
      savedAnalysis =
        (request.result as SavedAnalysis<T> | undefined) ?? null
    }

    request.onerror = () => {
      database.close()
      reject(
        request.error ?? new Error('Could not load saved analysis.'),
      )
    }

    transaction.oncomplete = () => {
      database.close()
      resolve(savedAnalysis)
    }

    transaction.onerror = () => {
      database.close()
      reject(
        transaction.error ?? new Error('Could not read saved analysis.'),
      )
    }

    transaction.onabort = () => {
      database.close()
      reject(
        transaction.error ?? new Error('Reading saved analysis was aborted.'),
      )
    }
  })
}

export async function clearAnalysis(): Promise<void> {
  const database = await openDatabase()

  return new Promise((resolve, reject) => {
    const transaction = database.transaction(STORE_NAME, 'readwrite')
    const store = transaction.objectStore(STORE_NAME)

    store.delete(ANALYSIS_KEY)

    transaction.oncomplete = () => {
      database.close()
      resolve()
    }

    transaction.onerror = () => {
      database.close()
      reject(
        transaction.error ?? new Error('Could not clear saved analysis.'),
      )
    }

    transaction.onabort = () => {
      database.close()
      reject(
        transaction.error ?? new Error('Clearing saved analysis was aborted.'),
      )
    }
  })
}