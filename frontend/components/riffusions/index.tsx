import React, { useEffect, useState } from "react";
import { GenreMapEntries } from "../../utils/GenreMap";
import { useRecoilState, useRecoilValue, useSetRecoilState } from "recoil";
import {
  canPostMusicState,
  musicalInstrumentsState,
  producingMusicState,
  trackAtomFamily,
  tracksInfoState,
} from "../../store/atoms";
import serverApi from "../../services/serverApi";
import { InstrumentsMapEntries } from "../../utils/InstrumentsMap";
import { toastAlert } from "../../utils/toastAlert";
import axios from "axios";
import ContentContainer from "../dashboard/library/contentContainer";
import CssSpinner from "../cssSpinner";
import MusicPlayer from "../dashboard/musicPlayer";

function Riffusions() {
  // let [producingMusic, setProducingMusic] = useState({
  //   title: null,
  //   tags: ["Happy", "Sad", "Energetic"],
  //   description: null,
  //   genre: "2",
  //   beat: 100,
  //   created_at: 1684127895274,
  //   music_source:
  //     "https://chatcomposer.s3.ap-northeast-2.amazonaws.com/origin-music/703.7256216117561_2023-05-15audio.wav",
  //   mixed_music_request: null,
  //   mixed_music_source: null,
  //   cover_request: "asdfsdfasdf",
  //   cover_source:
  //     "https://chatcomposer.s3.ap-northeast-2.amazonaws.com/album/710.1714783313225_2023-05-15image.png",
  // });
  const [producingMusic, setProducingMusic] =
    useRecoilState(producingMusicState);
  const [isLoading, setIsLoading] = useState(false);
  const setCanPost = useSetRecoilState(canPostMusicState);

  const [riffPrompt, setRiffPrompt] = useState("");
  const [instruementsArray, setInstruementsArray] = useRecoilState(
    musicalInstrumentsState,
  );
  const [mixedMusicUrl, setMixedMusicUrl] = useState("");
  const [tracksInfoArray, setTracksInfoArray] = useRecoilState(tracksInfoState);
  const [musicPrompt, setMusicPrompt] = useState("");
  const [mixedString, setMixedString] = useState("");
  const [mixedPrompt, setMixedPrompt] = useState("");
  const track0 = useRecoilValue(trackAtomFamily(0));
  const track1 = useRecoilValue(trackAtomFamily(1));
  const track2 = useRecoilValue(trackAtomFamily(2));
  const track3 = useRecoilValue(trackAtomFamily(3));

  useEffect(() => {
    [track0, track1, track2, track3].forEach((e, index) => {
      if (e?.musical_instrument) {
        setInstruementsArray((prev) => {
          if (!prev[index]) {
            const newArray = [...prev];
            newArray[index] = InstrumentsMapEntries[e?.musical_instrument][0];
            return newArray;
          } else return prev;
        });
        setTracksInfoArray((prev) => {
          if (!prev[index]) {
            const newArray = [...prev];
            newArray[index] = {
              request_description: e.request_description,
              response_description: e.response_description,
              transfer_date: e.transfer_date,
            };
            return newArray;
          } else return prev;
        });
      }
    });
  }, [track0, track1, track2, track3]);

  useEffect(() => {
    if (track0 && track0.request_description !== musicPrompt) {
      setMusicPrompt(track0.request_description);
    }
  }, [track0]);

  const onMixing = async () => {
    setIsLoading(true);
    const riffusionPrompt = await axios
      .post("/api/papago", { message: riffPrompt })
      .then((res) => res.data.message)
      .catch((err) => riffPrompt);
    //주석 console.log(riffusionPrompt);
    const data = {
      musicSource: producingMusic.music_source,
      genre: producingMusic.genre,
      moods: producingMusic.tags,
      instruements: instruementsArray,
      musicPrompt: musicPrompt,
      riffusionPrompt,
    };
    //주석 console.log(data);
    await serverApi
      .post("/produce/musics/riffusion", data)
      .then((res) => {
        setMixedString(res.data.mixed_music_wav);
      })
      .catch(async (err) => {
        //주석 console.log(JSON.stringify(err));
        const audio = new Audio("/dummy/riffusion/dummy.mp3"); // audio 객체 생성하기
        const response = await fetch(audio.src); // fetch API로 audio 객체의 URL을 blob으로 가져오기
        const blob = await response.blob(); // blob() 메서드로 응답을 blob으로 변환하기
        const reader = new FileReader(); // FileReader 객체 생성하기
        reader.readAsDataURL(blob); // blob을 base64로 읽기
        reader.onload = () => {
          //주석 console.log(reader.result);
          setMixedString(reader.result as string); // base64 문자열 저장하기
          setMixedPrompt(musicPrompt);
          setMusicPrompt("");
        };
        // };
      });
    setIsLoading(false);
    // setTimeout(()=>onSubmitHandler(), 100)
    // toastAlert(`음악 생성 완료!`);
    /*
{
	music_source : "www.naver.com",
	genre : "Pop",
	moods : ["Happy","Sad","Energetic"],
	instruements : ["piano", "guitar-acoustic"],
	music_prompt : "[instruement: piano]Music that's good to listen to on a rainy day",
	riffusion_prompt: "언젠가부터 힙합은 안멋져",
}
    */
  };
  function base64DecodeUnicode(str) {
    // Convert Base64 encoded bytes to percent-encoding, and then get the original string.
    const percentEncodedStr = atob(str)
      .split("")
      .map(function (c) {
        return "%" + ("00" + c.charCodeAt(0).toString(16)).slice(-2);
      })
      .join("");
    //주석 console.log(percentEncodedStr);
    return decodeURIComponent(percentEncodedStr);
  }

  const onSubmitHandler = async () => {
    const getAudioBlob = async () => {
      let base64String = mixedString.replace("data:audio/mpeg;base64,", "");

      // const base64String = mixedString;
      // //주석 console.log(base64String);
      // //주석 console.log(mixedString);
      if (!mixedString.startsWith("data")) {
        base64String = base64DecodeUnicode(mixedString);
      }

      // const decoded = atob(mixedString)
      const byteArray = Buffer.from(base64String, "base64");
      const blob = new Blob([byteArray], { type: "audio/mpeg" }); // Blob 객체 생성
      const file = new File([blob], "audio.mp3", { type: "audio/mpeg" }); // File 객체 생성

      return file;
    };

    const formData = new FormData();

    const audioFile = await getAudioBlob();
    formData.append("file", audioFile);

    await serverApi
      .post("/produce/musics/mixed", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      })
      .then((res) => {
        //주석 console.log("S3음악저장완료");
        //주석 console.log(res.data.source);
        setMixedMusicUrl(res.data.source);
        setProducingMusic((prev) => ({
          ...prev,
          mixed_music_source: res.data.source,
          mixed_music_request: mixedPrompt,
        }));
      });

    setCanPost(true);
    toastAlert(`음악 저장 완료!`);
    //   .post("/produce/cover", formData, {
    //     headers: { "Content-Type": "multipart/form-data" },
    //   })
    //   .then((res) => {
    //     setProducingMusic((prev) => {
    //       return {
    //         ...prev,
    //         cover_source: res.data.source,
    //         cover_request: selectedImgPrompt,
    //       };
    //     });
    //   })
    //   .catch((err) => {
    //     if (selectedImgURL.startsWith("/")) {
    //       setProducingMusic((prev) => {
    //         return {
    //           ...prev,
    //           cover_source: selectedImgURL,
    //           cover_request: selectedImgPrompt,
    //         };
    //       });
    //     }
    //   });

    // const  mixed_music_source = await serverApi.post("/produce/musics/mixed", )
  };
  useEffect(() => {
    if (mixedString) onSubmitHandler();
  }, [mixedString]);
  return (
    // <div className="text-pink-500">
    //   {JSON.stringify(instruementsArray)}
    //   {JSON.stringify(musicPrompt)}
    //   {mixedString}
    //   <div>

    //   </div>
    //   {mixedString ? (
    //     <>
    //       <audio src={mixedString} controls />
    //       <button onClick={onSubmitHandler}>저장하기</button>
    //     </>
    //   ) : (
    //     <></>
    //   )}
    //   {JSON.stringify(producingMusic)}
    //   <div>트랙인포</div>
    //   {JSON.stringify(tracksInfoArray)}
    // </div>
    <div className="w-full flex mt-10 mb-5">
      <div className="mx-auto w-full lg:max-w-[50%]">
        <h3 className="w-full text-center libray_h3 font-bold text-pink-500 text-2xl mb-[15px] mt-2">
          음악 믹싱하기
        </h3>
        <ContentContainer>
          <div className="w-full h-full flex">
            <input
              className="w-full bg-white-50 border border-pink-200 placeholder-gray-400 px-4 text-gray-700"
              value={riffPrompt}
              onChange={(e) => setRiffPrompt(e.target.value)}
              placeholder="어떻게 믹싱할까요?"
              autoFocus={true}
            ></input>
            <div className="w-full max-w-[220px] flex">
              <button
                type="button"
                onClick={onMixing}
                className={`inline-block text-sm px-4 py-2 w-full max-w-[220px]  leading-none border text-white border-pink-400  mx-1 bg-pink-500 ${
                  riffPrompt && !isLoading
                    ? "hover:border-transparent hover:text-pink-500 hover:bg-pink-200"
                    : "opacity-50"
                }`}
                disabled={!riffPrompt || isLoading}
              >
                {isLoading ? (
                  <div className="flex justify-center">
                    <CssSpinner></CssSpinner>
                  </div>
                ) : (
                  "믹싱하기"
                )}
              </button>
            </div>
          </div>
          <div className="w-full flex">
            {/* <button
              type="button"
              onClick={onSubmitHandler}
              className={`inline-block mx-auto mt-5 mb-3 text-sm px-4 py-2 w-full max-w-[220px]  leading-none border text-white border-pink-400  bg-pink-500 ${
                mixedString
                  ? "hover:border-transparent hover:text-pink-500 hover:bg-pink-200"
                  : "opacity-50"
              }`}
              disabled={!mixedString}
            >
              믹싱된 음악 저장하기
            </button> */}
            {mixedMusicUrl ? (
              <audio
                src={mixedMusicUrl}
                controls
                className="mx-auto my-4"
              ></audio>
            ) : (
              <></>
            )}
          </div>
        </ContentContainer>
      </div>
    </div>
  );
}

export default Riffusions;
