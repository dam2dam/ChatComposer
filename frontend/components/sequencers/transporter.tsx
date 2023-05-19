import React, { useEffect, useMemo, useState } from "react";
import SequencerControlBox from "./sequencerControlBox";
import { useRecoilState, useRecoilValue, useSetRecoilState } from "recoil";
import {
  blobAudioState,
  lastScheduleTimeState,
  sheduleArrayState,
  trackAtomFamily,
} from "../../store/atoms";
import * as Tone from "tone";
import {
  InstrumentsMapEntries,
  InstrumentsUrl,
} from "../../utils/InstrumentsMap";
import CssSpinner from "../cssSpinner";
import { Transport } from "tone/build/esm/core/clock/Transport";
// import WavRecorder from "webm-to-wav-converter/types/WavRecorder";
// import getWaveBlob from "webm-to-wav-converter/types/wavBlobUtil";
import { getWaveBlob } from "webm-to-wav-converter";
import { toastAlert } from "../../utils/toastAlert";

function Transporter({ trackId }) {
  const [track, setTrack] = useRecoilState(trackAtomFamily(trackId));
  const setSheduleArrayAtom = useSetRecoilState(sheduleArrayState);
  const setAudioState = useSetRecoilState(blobAudioState);
  const [lastScheduleTime, setLastScheduleTime] = useRecoilState(
    lastScheduleTimeState,
  );
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (!track?.midi_description || trackId) return;
    const notes = JSON.parse(track?.midi_description);
    if (!notes.length) {
      setTrack(() => null);
      toastAlert("새로운 프롬프트로 다시 시도해주세요");
    }
    notes
      .sort((a, b) => a[2] - b[2])
      .forEach((e, i) => {
        if (i >= notes.length - 1) {
          setLastScheduleTime(e[2]);
        }
      });
  }, [track?.midi_description]);

  if (!track?.midi_description) {
    return <div></div>;
  }

  const samplerOptions =
    InstrumentsUrl[InstrumentsMapEntries[track?.musical_instrument][0]];

  const onShedule = (transport: Transport, recorder: Tone.Recorder) => {
    const notes = JSON.parse(track?.midi_description);
    sampler.connect(recorder);
    notes
      .sort((a, b) => a[2] - b[2])
      .forEach((e, i) => {
        // transport.schedule로 바꿈

        //피아노의 마지막 스케줄보다 빨리 시작하는 이벤트만 등록함.
        if (e[2] <= lastScheduleTime) {
          transport.schedule((time) => {
            // 콜백 함수에서 time을 인자로 받음
            sampler.triggerAttackRelease(e[0], e[1], time);
          }, e[2]); // 스케줄링할 시간은 e[2]
        }

        if (trackId === 0 && i === notes.length - 1) {
          //만약, 피아노이고 마지막 노트라면 2박자 후에 노래를 정지한다.
          //노래를 정지하는 이벤트
          transport.schedule((time) => {
            transport.stop();
            recorder.stop().then(async (blob) => {
              const wavBlog = await getWaveBlob(blob, false, {
                sampleRate: 44100,
              });
              //주석 console.log(wavBlog);
              setAudioState(wavBlog);
              // // blob을 Multipartfile로 변환
              // const formData = new FormData();
              // formData.append("audio", blob);
              // // formData를 서버에 전송
              // fetch("/upload", {
              //   method: "POST",
              //   body: formData,
              // });
            });
          }, e[2] + 2);
        }
      });
  };

  const sampler = new Tone.Sampler({
    ...samplerOptions,
    onload: () => {
      setIsLoading(false);
      sampler.toDestination();
      setSheduleArrayAtom((prev) => {
        const newArray = [...prev];
        newArray[trackId] = onShedule;
        return newArray;
      });
    },
  });

  const onPlay = () => {
    const notes = JSON.parse(track?.midi_description);

    const now = Tone.now();
    Tone.Transport.bpm.value = 100;

    notes.forEach((e, i) => {
      sampler.triggerAttackRelease(e[0], e[1], now + e[2]);
    });
  };

  return (
    <div className="w-full flex border-2 border-pink-500">
      <div className="mx-auto my-auto">
        {isLoading ? <CssSpinner /> : <SequencerControlBox onPlay={onPlay} />}
      </div>
    </div>
  );
}

export default React.memo(Transporter);
