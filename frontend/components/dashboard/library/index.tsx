import React, { useEffect } from "react";
import LibraryUpper from "./upper";
import Albums from "./albums";
import Trending from "./trending";
import Favorite from "./favorite";
import { useRecoilValue } from "recoil";
import { genreDataState, selectedGenreState } from "../../../store/atoms";
import { IconMusic } from "@tabler/icons-react";
import Link from "next/link";

function Library() {
  const genreSet = useRecoilValue(genreDataState);
  const selectedGenre = useRecoilValue(selectedGenreState);
  // useEffect(() => {
  //   //주석 console.log(genreSet);
  // }, [genreSet]);
  return (
    <div className="mp_library relative w-full h-full">
      {/* <LibraryUpper /> */}
      <div className="h-[20px] w-full"></div>
      {/* <div className="relative w-full mx-auto border mt-5 py-1 text-center"> */}
      <div className="relative w-full mx-auto border rounded-lg mt-5 p-1 text-center">
        <Link
          href={"/produce"}
          className="bg-black absolute rounded top-0 left-0 w-full h-full z-[6] opacity-0 hover:opacity-60"
        >
          <IconMusic
            className="library_album_covers_img absolute z-[4] w-[100px] top-[50%] left-1/2 translate-x-[-50%] translate-y-[-50%] 가운데에넣기"
            size={48}
            strokeWidth={2}
            color={"#ffffff"}
          />
          <div className="text-pink-100 absolute text-2xl z-[4] w-[150px] top-[60%] left-1/2 translate-x-[-50%] translate-y-[-50%]">
            클릭하여 이동
          </div>
        </Link>
        {/* <div className="text-white absolute text-4xl top-[4vh] md:right-[2vw] xl:right-[9vw] bg-black/40 text-right p-5 hidden xl:block pb-[8%] pl-[10%] pt-[6%] pr-[3%]">
          <b className="text-pink-100">영준몬</b>이 선택한
          <div className="text-5xl mt-2">
            단 하나의 <span className="text-pink-200">작곡앱</span>
          </div>
          <div className="text-xl mt-2 text-orange-100 ">
            "저는 항상 ChatComposer를 통해 영감을 얻죠"
            <br />
            <span className="text-white">
              - 음악의 신, <b>영준몬</b>
            </span>
          </div>
        </div>
        <h3 className="libray_h3 text-4xl font-bold text-[17px] mb-[15px] ml-6 absolute top-3/4 left-1/2 lg:top-[77%] lg:left-[72%] translate-x-[-50%] translate-y-[-50%] bg-orange-600 text-white px-20 py-2 rounded-xl shadow-sm">
          지금 작곡하기
        </h3> */}
        {/* <img src="banner.png" className="max-w-sm mx-auto" /> */}
        {/* <img src="main-banner.jpg" className="mx-auto" /> */}
        <img src="main-banner.jpg" className="mx-auto rounded-lg" />
      </div>
      <Albums />
      <Trending genre={selectedGenre} />
      <Favorite />
      {genreSet.map((e, index) => {
        return <Favorite genre={e} key={e + index} />;
      })}
      <div className="h-[300px] w-full"></div>
    </div>
  );
}

export default Library;
