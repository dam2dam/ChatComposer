import React, { useEffect, useState } from "react";
import { useRecoilState, useRecoilValue, useSetRecoilState } from "recoil";
import {
  coversState,
  producingMusicState,
  selectedCoverPromptState,
  selectedCoverState,
  trackAtomFamily,
} from "../../store/atoms";
import serverApi from "../../services/serverApi";
import axios from "axios";
import { toastAlert } from "../../utils/toastAlert";
import CssSpinner from "../cssSpinner";
import ContentContainer from "../dashboard/library/contentContainer";

function CoverGens() {
  const [coversArray, setCoversArray] = useRecoilState(coversState);
  const [selectedImgURL, setSelectedImgURL] =
    useRecoilState(selectedCoverState);
  const [selectedImgPrompt, setSelectedPrompt] = useRecoilState(
    selectedCoverPromptState,
  );
  const [isImageLoading, setIsImageLoading] = useState(false);
  const setProducingMusic = useSetRecoilState(producingMusicState);

  const retrieveCovers = async (coverRequest: string) => {
    const requestMessage = await axios
      .post("/api/papago", { message: coverRequest })
      .then((res) => res.data.message)
      .catch((err) => coverRequest);
    //주석 console.log(requestMessage);
    setIsImageLoading(true);
    const coverImageUrl = await serverApi
      .get(`/produce/cover?cover-request=${requestMessage}`)
      // const coverImageUrl = await axios
      //   .get(`/api/produce/cover?cover-request=${requestMessage}`)
      .then((res) => {
        const imageURL = `data:image/png;base64,${res.data.cover}`;
        // const byteArray = new Uint8Array(res.data.cover);
        // const blob = new Blob([byteArray], { type: "image/png" });
        // const imageURL = URL.createObjectURL(blob);
        // const image = new Image();
        // image.src = imageURL;
        // return Promise.reject("");
        return imageURL;
      })
      .catch((err) => {
        //주석 console.log(JSON.stringify(err));
        // const image = new Image();
        // image.src = `/dummy/covers/cover${
        //   Math.floor(Math.random() * 11) + 1
        // }.png`;
        // const imageURL = `/dummy/covers/cover${
        //   Math.floor(Math.random() * 11) + 1
        // }.png`;

        // return imageURL;
        return new Promise((resolve) => {
          setTimeout(() => {
            const imageURL = `/dummy/covers/cover${
              Math.floor(Math.random() * 11) + 1
            }.png`;
            resolve(imageURL);
          }, 1500); // 2초 후에 값을 반환
        });
      });
    setSelectedImgURL(coverImageUrl as string);
    setSelectedPrompt(coverRequest);
    setIsImageLoading(false);
    toastAlert(`앨범 커버 생성 완료!`);
    return coverImageUrl;
  };

  const [coverRequest, setCoverRequest] = useState("");
  const onRetrieveCovers = () => {
    if (!coverRequest) return;
    retrieveCovers(coverRequest).then((image) =>
      setCoversArray((prev) => [...prev, [image, coverRequest]]),
    );
    setCoverRequest("");
  };

  return (
    <div className="w-full max-w-[700px] mx-auto">
      <h3 className="w-full text-center libray_h3 font-bold text-pink-500 text-2xl mb-[15px] mt-2">
        앨범 커버 생성
      </h3>
      <ContentContainer>
        <div
          className={`h-8 align-top ${
            coversArray.length < 5 ? "" : "hidden"
          } flex `}
        >
          <input
            className="h-full bg-pink-50/60 w-full max-w-[560px] placeholder-gray-400 px-4 text-gray-700"
            placeholder="어떤 앨범 커버를 만들까요?"
            value={coverRequest}
            onChange={(e) => setCoverRequest(e.target.value)}
          />
          {isImageLoading ? (
            <button
              type="button"
              className={`h-full inline-block px-4 w-full max-w-[130px]  leading-none text-white hover:border-transparent hover:text-pink-500 hover:bg-pink-200 bg-pink-500 opacity-50
          }`}
              disabled
            >
              <div className="flex justify-center">
                <CssSpinner />
              </div>
            </button>
          ) : (
            <button
              type="button"
              onClick={onRetrieveCovers}
              className={`h-full inline-block px-4 w-full max-w-[130px]  leading-none text-white hover:border-transparent hover:text-pink-500 hover:bg-pink-200 bg-pink-500 ${
                coverRequest ? "" : "opacity-50"
              }`}
              disabled={!coverRequest}
            >
              {coverRequest ? "생성하기" : "입력해주세요"}
            </button>
          )}
        </div>
        <div className="flex">
          {selectedImgURL ? (
            <div className="text-center text-pink-700">
              <img
                className="max-w-[200px]  mx-auto aspect-square"
                src={selectedImgURL}
              />
              선택된 이미지
            </div>
          ) : (
            <></>
          )}
          <div className="flex">
            {coversArray.map(([url, prompt], index) => {
              return (
                <img
                  key={index + url}
                  src={url}
                  className="max-w-[100px] max-h-[100px] aspect-square"
                  onClick={() => {
                    setSelectedImgURL(url);
                    setSelectedPrompt(prompt);
                  }}
                />
              );
            })}
          </div>
        </div>
      </ContentContainer>
    </div>
  );
}

export default CoverGens;

// {/* <div className="w-full h-full flex">
// <input
//   className="w-4/5 bg-gray-700 placeholder-gray-300 px-4 text-white"
//   value={input}
//   onChange={onChangeHandler}
//   placeholder="만들고 싶은 음악을 묘사해보세요!"
//   autoFocus={true}
// ></input>

// {/* <button
//   type="button"
//   onClick={onSubmitHandler}
//   className={`w-1/5 ${
//     !isLoading && track?.musical_instrument !== null && input
//       ? ""
//       : "opacity-50"
//   }`}
//   disabled={!(!isLoading && track?.musical_instrument !== null && input)}
// >
//   {isLoading ? (
//     <div className="flex justify-center">
//       <CssSpinner />
//     </div>
//   ) : (
//     <>음악 생성하기2</>
//   )}
// </button> */}
{
  /* <div className="w-full max-w-[220px] flex">
  <button
    type="button"
    onClick={onSubmitHandler}
    className={`inline-block text-sm px-4 py-2 w-full max-w-[220px]  leading-none border text-white border-pink-400 hover:border-transparent hover:text-pink-500 hover:bg-pink-100/50 mx-1 bg-pink-500 ${
      !isLoading && track?.musical_instrument !== null && input
        ? ""
        : "opacity-50"
    }`}
    disabled={!(!isLoading && track?.musical_instrument !== null && input)}
  >
    {isLoading ? (
      <div className="flex justify-center">
        <CssSpinner />
      </div>
    ) : (
      <>
        {!(!isLoading && track?.musical_instrument !== null && input)
          ? "입력하세요"
          : "음악 작곡하기"}
      </>
    )}
  </button>
  <Transporter trackId={trackId} />
</div>
</div>  */
}
