const DATABASE_NAME = "prooflens-assets";
const DATABASE_VERSION = 1;
const STORE_NAME = "model-assets";
const MODEL_KEY = "detector-model";

export interface ModelRecord {
  key: string;
  modelId: string;
  sha256: string;
  bytes: ArrayBuffer;
  installedAt: string;
}

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DATABASE_NAME, DATABASE_VERSION);
    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(STORE_NAME)) {
        database.createObjectStore(STORE_NAME, { keyPath: "key" });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("Could not open model storage"));
  });
}

export async function readStoredModel(): Promise<ModelRecord | undefined> {
  const database = await openDatabase();
  try {
    return await new Promise((resolve, reject) => {
      const request = database.transaction(STORE_NAME, "readonly").objectStore(STORE_NAME).get(MODEL_KEY);
      request.onsuccess = () => resolve(request.result as ModelRecord | undefined);
      request.onerror = () => reject(request.error ?? new Error("Could not read model storage"));
    });
  } finally {
    database.close();
  }
}

export async function storeModel(record: Omit<ModelRecord, "key">): Promise<void> {
  const database = await openDatabase();
  try {
    await new Promise<void>((resolve, reject) => {
      const transaction = database.transaction(STORE_NAME, "readwrite");
      const request = transaction.objectStore(STORE_NAME).put({ ...record, key: MODEL_KEY } satisfies ModelRecord);
      request.onerror = () => reject(request.error ?? new Error("Could not store model"));
      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(transaction.error ?? new Error("Could not commit model storage"));
      transaction.onabort = () => reject(transaction.error ?? new Error("Model storage was aborted"));
    });
  } finally {
    database.close();
  }
}

export async function deleteStoredModel(): Promise<void> {
  const database = await openDatabase();
  try {
    await new Promise<void>((resolve, reject) => {
      const transaction = database.transaction(STORE_NAME, "readwrite");
      transaction.objectStore(STORE_NAME).delete(MODEL_KEY);
      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(transaction.error ?? new Error("Could not clear model storage"));
    });
  } finally {
    database.close();
  }
}
