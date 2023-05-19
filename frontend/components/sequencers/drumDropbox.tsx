import { Transport } from "tone/build/esm/core/clock/Transport";
import * as Tone from "tone";
import CssSpinner from "../cssSpinner";
import { useState } from "react";
import { DrumArray, InstrumentsUrl } from "../../utils/InstrumentsMap";
import { useRecoilValue, useSetRecoilState } from "recoil";
import {
  blobAudioState,
  lastScheduleTimeState,
  sheduleArrayState,
} from "../../store/atoms";
import { getWaveBlob } from "webm-to-wav-converter";
import { IconPlayerPlayFilled } from "@tabler/icons-react";

function DrumDropbox() {
  const drumBeatsArray = [
    "8bit-south-hiphop",
    "8bit-oldschool",
    "8bit-dubstep",
    "8bit-house",
  ];
  const [isLoading, setIsLoading] = useState(true);
  const setSheduleArrayAtom = useSetRecoilState(sheduleArrayState);
  const lastScheduleTime = useRecoilValue(lastScheduleTimeState);
  const setAudioState = useSetRecoilState(blobAudioState);

  const [selectedBeats, setSelectedBeats] = useState(4);

  const sampler = new Tone.Sampler({
    ...InstrumentsUrl.drum,
    onload: () => {
      setIsLoading(false);
      sampler.toDestination();
      setSheduleArrayAtom((prev) => {
        const newArray = [...prev];
        newArray[3] = onShedule;
        return newArray;
      });
    },
  });

  const onShedule = (transport: Transport, recorder: Tone.Recorder) => {
    const beatsName = drumBeatsArray[selectedBeats];
    const notes = DrumArray[beatsName].map((e) => [e[0], e[1], e[2] / 2]);
    sampler.connect(recorder);
    notes
      .sort((a, b) => a[2] - b[2])
      .forEach((e, i) => {
        if (e[2] <= lastScheduleTime) {
          transport.schedule((time) => {
            sampler.triggerAttackRelease(e[0], e[1], time);
          }, e[2]);
        }
      });
  };
  const onChangeHandler = (e) => {
    setSelectedBeats(e.target.value);
  };

  const onPlay = () => {
    sampler.releaseAll();
    const beatsName = drumBeatsArray[selectedBeats];
    const notes = DrumArray[beatsName]
      .slice(0, 30)
      .map((e) => [e[0], e[1], e[2] / 2]);

    const now = Tone.now();
    Tone.Transport.bpm.value = 100;

    notes.forEach((e, i) => {
      sampler.triggerAttackRelease(e[0], e[1], now + e[2]);
    });
  };
  return (
    <div className="GenreDropboxContainer flex">
      {/* https://devdojo.com/zoltan/tailwind-css-select */}
      <select
        id="beats"
        // className="bg-gray-700 border border-gray-600 text-white text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5 placeholder-gray-400"
        className="bg-pink-700 border border-pink-600 text-pink-50 text-sm focus:ring-pink-500 focus:border-pink-500 block w-full p-2.5 placeholder-pink-400"
        defaultValue="default"
        onChange={onChangeHandler}
      >
        <option value="default" disabled className="hidden">
          비트를 선택하세요
        </option>
        {drumBeatsArray.map((element, index) => {
          return (
            <option key={element + index} value={index}>
              {element}
            </option>
          );
        })}
      </select>
      <div className="w-1/6 flex border border-pink-600">
        {isLoading ? (
          <div className="mx-auto my-auto ">
            <CssSpinner />
          </div>
        ) : (
          <div
            role="button"
            className={`mx-auto my-auto text-pink-500 ${
              selectedBeats < 4 ? "" : "hidden"
            }`}
            onClick={selectedBeats < 4 ? onPlay : () => {}}
          >
            <IconPlayerPlayFilled strokeWidth={2} color={"pink"} />
          </div>
        )}
      </div>
    </div>
  );
}

export default DrumDropbox;
