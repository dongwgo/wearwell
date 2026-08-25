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
    putLook(record) { return withStore("looks", "readwrite", store => store.put(record)); },
    getLook(id) { return withStore("looks", "readonly", store => store.get(id)); }
  };

  window.WearwellDB = api;
})();
