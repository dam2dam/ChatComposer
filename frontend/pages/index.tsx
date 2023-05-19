import { useEffect, useState } from "react";
import Library from "../components/dashboard/library";
import MusicPlayer from "../components/dashboard/musicPlayer";
import {
  prefetchListGenreMusicsQuery,
  prefetchListTagMusicsQuery,
} from "../services/musics";
import { GenreMapEntries, TagMapEntries } from "../utils/GenreMap";
import { useRouter } from "next/router";
import { useRecoilState, useRecoilValue } from "recoil";
import { authTokenState } from "../store/atoms";
import Link from "next/link";
import serverApi from "../services/serverApi";
import Reset from "./reset";

export default function Home() {
  const [isPrefetched, setIsPrefetched] = useState(false);
  const [opacityClassname, setOpacityClassname] = useState("");
  const [isMounted, setIsMounted] = useState(false);
  const [authToken, setAuthToken] = useRecoilState(authTokenState);
  const router = useRouter();
  const [oauthUrl, setOauthUrl] = useState(
    "https://k8a504.p.ssafy.io/api/oauth2/authorization/google?redirect_uri=https://k8a504.p.ssafy.io/oauth/redirect",
  );

  useEffect(() => {
    setIsMounted(true);
  }, []);
  // const checkAuth = () => {
  //   if (authToken) {
  //     serverApi
  //       .get("/users")
  //       .then(() => {})
  //       .catch((err) => {
  //         //주석 console.log(err.status);
  //         setAuthToken(null);
  //         router.push("/");
  //       });
  //   }
  // };
  useEffect(() => {
    //데이터를 프리페치하여 메인 페이지의 로딩 시간을 줄입니다.
    Promise.allSettled([
      ...GenreMapEntries.map((e) => prefetchListGenreMusicsQuery(e[0])),
      ...TagMapEntries.map((e) => prefetchListTagMusicsQuery(e[0])),
    ]).then(() => {
      //prefetch가 끝났으면 1초 후 isPrefetched는 true입니다.
      setTimeout(() => setIsPrefetched(true), 1000);
    });
  }, []);

  // useEffect(() => {
  //   if (authToken) {
  //     serverApi
  //       .get("/users")
  //       .then(() => {})
  //       .catch((err) => {
  //         //주석 console.log()
  //         setAuthToken(null);
  //         router.push("/");
  //       });
  //   }
  // }, [isPrefetched]);

  useEffect(() => {
    if (isPrefetched && authToken) {
      serverApi
        .get("/users")
        .then(() => {
          setOpacityClassname("opacity-0");
          setTimeout(() => router.push("/main"), 300);
        })
        .catch((err) => {
          if (err.response.status === 401) {
            setAuthToken((prev) => {
              localStorage.clear();
              return null;
            });
            setTimeout(() => router.push("/"), 300);
          }
        });
    }
  }, [isPrefetched, authToken]);

  return (
    <div className="h-screen overflow-hidden flex items-center justify-center">
      <div className="relative overflow-hidden">
        <div className={`transition-all duration-200  ${opacityClassname}`}>
          <div className="mx-auto max-w-5xl lg:px-8">
            <div className="lg:grid lg:grid-cols-2 lg:gap-8">
              <div className="mx-auto max-w-md px-4 text-center sm:max-w-2xl sm:px-6 lg:flex lg:items-center lg:px-0 lg:text-left">
                <div className="lg:py-24">
                  <h1 className="mt-4 text-4xl font-bold tracking-tight text-black sm:mt-5 sm:text-6xl lg:mt-6 xl:text-6xl">
                    <span className="block text-pink-500">ChatComposer</span>
                    {/* <span className="block text-black">음악을 작곡하세요</span> */}
                  </h1>
                  <p className="mt-3 text-base text-gray-400 sm:mt-5 sm:text-xl lg:text-lg xl:text-xl">
                    지금 구글로 로그인하고
                    <br />
                    인공지능과 함께 음악을 작곡하세요.
                  </p>
                  <div className="mt-10 sm:mt-12">
                    {/* This is a working waitlist form. Create your access key from https://web3forms.com/s to setup.  */}
                    <form
                      className="sm:mx-auto sm:max-w-xl lg:mx-0"
                      action="https://api.web3forms.com/submit"
                    >
                      <div className="sm:flex">
                        <input
                          type="hidden"
                          name="access_key"
                          defaultValue="YOUR_ACCESS_KEy_HERE"
                        />
                        <input
                          type="hidden"
                          name="subject"
                          defaultValue="New Waitlist"
                        />
                        <div className="mt-3 sm:mt-0 sm:ml-3">
                          <Link
                            href={oauthUrl}
                            rel="noopener noreferrer"
                            className={authToken || !isMounted ? "hidden" : ""}
                          >
                            <button
                              type="button"
                              className="block w-full rounded-md bg-pink-500 py-3 px-4 font-medium text-white shadow hover:bg-pink-400 focus:outline-none focus:ring-2 focus:ring-blue-300 focus:ring-offset-2 focus:ring-offset-gray-900"
                            >
                              Google로 로그인하기
                            </button>
                          </Link>
                        </div>
                      </div>
                    </form>
                  </div>
                </div>
              </div>
              <img
                className="max-h-[50%] m-auto hidden lg:block"
                // src="https://user-images.githubusercontent.com/1884712/202186141-9f8a93e1-7743-459a-bc95-b1d826931624.png"
                src="banner.png"
                alt="대충 서비스 소개 이미지"
              />
            </div>
          </div>
        </div>
      </div>
      <Reset />
    </div>
  );
}
