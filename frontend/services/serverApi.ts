import axios from "axios";

const serverApi = axios.create({
  baseURL: "https://k8a504.p.ssafy.io/api",
});

serverApi.interceptors.request.use((config) => {
  if (config.url.startsWith("/oauth2")) return config;

  const getToken = () => {
    let token = "";
    const recoilState = localStorage.getItem("recoil-persist");
    if (recoilState) {
      const newToken = JSON.parse(recoilState)?.AuthTokenState;
      if (newToken) {
        token = newToken;
      }
    }
    return token;
  };

  if (config.method === "get" && config.url.startsWith("/musics")) {
    const authToken = getToken();
    if (!authToken) return config;
  }

  config.headers["Authorization"] = `Bearer ${getToken()}`;
  return config;
});

export default serverApi;
