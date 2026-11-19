const DATABASE_NAME = 'nav-fund-analytics'
const DATABASE_VERSION = 2
const ANALYSIS_STORE = 'analysis'
const ASSISTANT_STORE = 'assistant-chat'
const ANALYSIS_KEY = 'latest-analysis'
const ASSISTANT_CHAT_KEY = 'conversation'

export type SavedAnalysis<T> = {
  results: T
  selectedFundIndex: number
  savedAt: string
}

export type SavedAssistantChat<T> = {
  messages: T[]
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

      if (!database.objectStoreNames.contains(ANALYSIS_STORE)) {
        database.createObjectStore(ANALYSIS_STORE)
      }

      if (!database.objectStoreNames.contains(ASSISTANT_STORE)) {
        database.createObjectStore(ASSISTANT_STORE)
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
        request.error ?? new Error('Could not open application storage.'),
      )
    }

    request.onblocked = () => {
      reject(
        new Error('Application storage is blocked by another browser tab.'),
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
    const transaction = database.transaction(ANALYSIS_STORE, 'readwrite')

    const savedAnalysis: SavedAnalysis<T> = {
      results,
      selectedFundIndex,
      savedAt: new Date().toISOString(),
    }

    transaction.objectStore(ANALYSIS_STORE).put(
      savedAnalysis,
      ANALYSIS_KEY,
    )

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
    const transaction = database.transaction(ANALYSIS_STORE, 'readonly')
    const request = transaction.objectStore(ANALYSIS_STORE).get(ANALYSIS_KEY)

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
    const transaction = database.transaction(ANALYSIS_STORE, 'readwrite')

    transaction.objectStore(ANALYSIS_STORE).delete(ANALYSIS_KEY)

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

export async function saveAssistantChat<T>(
  messages: T[],
): Promise<void> {
  const database = await openDatabase()

  return new Promise((resolve, reject) => {
    const transaction = database.transaction(ASSISTANT_STORE, 'readwrite')

    const savedChat: SavedAssistantChat<T> = {
      messages,
      savedAt: new Date().toISOString(),
    }

    transaction.objectStore(ASSISTANT_STORE).put(
      savedChat,
      ASSISTANT_CHAT_KEY,
    )

    transaction.oncomplete = () => {
      database.close()
      resolve()
    }

    transaction.onerror = () => {
      database.close()
      reject(
        transaction.error ?? new Error('Could not save assistant chat.'),
      )
    }

    transaction.onabort = () => {
      database.close()
      reject(
        transaction.error ?? new Error('Saving assistant chat was aborted.'),
      )
    }
  })
}

export async function loadAssistantChat<T>(): Promise<SavedAssistantChat<T> | null> {
  const database = await openDatabase()

  return new Promise((resolve, reject) => {
    const transaction = database.transaction(ASSISTANT_STORE, 'readonly')
    const request = transaction
      .objectStore(ASSISTANT_STORE)
      .get(ASSISTANT_CHAT_KEY)

    let savedChat: SavedAssistantChat<T> | null = null

    request.onsuccess = () => {
      savedChat =
        (request.result as SavedAssistantChat<T> | undefined) ?? null
    }

    request.onerror = () => {
      database.close()
      reject(
        request.error ?? new Error('Could not load assistant chat.'),
      )
    }

    transaction.oncomplete = () => {
      database.close()
      resolve(savedChat)
    }

    transaction.onerror = () => {
      database.close()
      reject(
        transaction.error ?? new Error('Could not read assistant chat.'),
      )
    }

    transaction.onabort = () => {
      database.close()
      reject(
        transaction.error ?? new Error('Reading assistant chat was aborted.'),
      )
    }
  })
}

export async function clearAssistantChat(): Promise<void> {
  const database = await openDatabase()

  return new Promise((resolve, reject) => {
    const transaction = database.transaction(ASSISTANT_STORE, 'readwrite')

    transaction.objectStore(ASSISTANT_STORE).delete(ASSISTANT_CHAT_KEY)

    transaction.oncomplete = () => {
      database.close()
      resolve()
    }

    transaction.onerror = () => {
      database.close()
      reject(
        transaction.error ?? new Error('Could not clear assistant chat.'),
      )
    }

    transaction.onabort = () => {
      database.close()
      reject(
        transaction.error ?? new Error('Clearing assistant chat was aborted.'),
      )
    }
  })
}