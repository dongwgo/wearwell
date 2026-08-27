(function () {
  const DB_NAME = "oneulout-fashion";
  const DB_VERSION = 2;
  let openPromise;

  function open() {
    if (openPromise) return openPromise;
    openPromise = new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, DB_VERSION);
      request.onupgradeneeded = () => {
        const database = request.result;
        if (!database.objectStoreNames.contains("garments")) database.createObjectStore("garments", { keyPath: "id" });
        // looks는 이전 버전과의 호환을 위해 유지한다. 공용 룩북 분석은 저장소의
        // assets/influencer-data.js에서 불러오며 신규 브라우저에는 중복 저장하지 않는다.
        if (!database.objectStoreNames.contains("looks")) database.createObjectStore("looks", { keyPath: "id" });
        // SigLIP 벡터 캐시. 옷장 한 벌의 사진은 바뀌지 않으므로 한 번 만든 벡터를
        // 새로고침 너머까지 들고 있는다 — 200벌을 매번 다시 인코딩하지 않기 위해서다.
        if (!database.objectStoreNames.contains("embeddings")) database.createObjectStore("embeddings", { keyPath: "id" });
      };
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
    return openPromise;
  }

  async function withStore(name, mode, action) {
    const database = await open();
    return new Promise((resolve, reject) => {
      const transaction = database.transaction(name, mode);
      const store = transaction.objectStore(name);
      const request = action(store);
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  const api = {
    putGarment(record) { return withStore("garments", "readwrite", store => store.put(record)); },
    getGarment(id) { return withStore("garments", "readonly", store => store.get(id)); },
    getAllGarments() { return withStore("garments", "readonly", store => store.getAll()); },
    deleteGarment(id) { return withStore("garments", "readwrite", store => store.delete(id)); },
    putLook(record) { return withStore("looks", "readwrite", store => store.put(record)); },
    getLook(id) { return withStore("looks", "readonly", store => store.get(id)); },
    getAllLooks() { return withStore("looks", "readonly", store => store.getAll()); },
    deleteLook(id) { return withStore("looks", "readwrite", store => store.delete(id)); },
    putEmbedding(record) { return withStore("embeddings", "readwrite", store => store.put(record)); },
    getAllEmbeddings() { return withStore("embeddings", "readonly", store => store.getAll()); },
    clearEmbeddings() { return withStore("embeddings", "readwrite", store => store.clear()); }
  };

  window.WearwellDB = api;
})();
