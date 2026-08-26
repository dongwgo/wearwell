(function () {
  const DB_NAME = "oneulout-fashion";
  const DB_VERSION = 1;
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
    getLook(id) { return withStore("looks", "readonly", store => store.get(id)); }
  };

  window.WearwellDB = api;
})();
