import { useRecoilState } from "recoil";
import Library from "../components/dashboard/library";
import MusicPlayer from "../components/dashboard/musicPlayer";
import { authTokenState } from "../store/atoms";
import { useRouter } from "next/router";
import { useEffect } from "react";
import serverApi from "../services/serverApi";
import { toastAlert } from "../utils/toastAlert";
import Reset from "./reset";

export default function Main() {
  const [authToken, setAuthToken] = useRecoilState(authTokenState);
  const router = useRouter();
  useEffect(() => {
    if (authToken) {
      serverApi.get("/users").catch((err) => {
        if (err.response.status === 401) {
          toastAlert("다시 로그인해주세요.");
          setAuthToken((prev) => {
            localStorage.clear();
            return null;
          });
          setTimeout(() => router.push("/"), 300);
        }
      });
    }
  }, [authToken]);

  return (
    <>
      <div className="">
        <Library />
        {/* <MusicPlayer /> */}
      </div>
      <Reset />
    </>
  );
}
