import React, { useEffect } from "react";
import { useResetRecoilState } from "recoil";
import {
  CoverGenHeightState,
  blobAudioState,
  canPostMusicState,
  coversState,
  lastScheduleTimeState,
  musicalInstrumentsState,
  prevDataState,
  producingMusicState,
  selectedCoverPromptState,
  selectedCoverState,
  sheduleArrayState,
  trackAtomFamily,
  trackIdsState,
  tracksInfoState,
} from "../store/atoms";
import { useRouter } from "next/router";

function Reset() {
  const resetProducingMusicState = useResetRecoilState(producingMusicState);
  const resetCoverGenHeightState = useResetRecoilState(CoverGenHeightState);
  const resetSheduleArrayState = useResetRecoilState(sheduleArrayState);
  const resetPrevDataState = useResetRecoilState(prevDataState);
  const resetLastScheduleTimeState = useResetRecoilState(lastScheduleTimeState);
  const resetBlobAudioState = useResetRecoilState(blobAudioState);
  const resetCoversState = useResetRecoilState(coversState);
  const resetSelectedCoverState = useResetRecoilState(selectedCoverState);
  const resetSelectedCoverPromptState = useResetRecoilState(
    selectedCoverPromptState,
  );
  const resetTrackIdsState = useResetRecoilState(trackIdsState);
  const resetTracksInfoState = useResetRecoilState(tracksInfoState);
  const resetMusicalInstrumentsState = useResetRecoilState(
    musicalInstrumentsState,
  );
  const resetCanPostMusicState = useResetRecoilState(canPostMusicState);
  const resetTrack0 = useResetRecoilState(trackAtomFamily(0));
  const resetTrack1 = useResetRecoilState(trackAtomFamily(1));
  const resetTrack2 = useResetRecoilState(trackAtomFamily(2));
  const resetTrack3 = useResetRecoilState(trackAtomFamily(3));
  const resetTrack4 = useResetRecoilState(trackAtomFamily(4));

  const router = useRouter();
  const resetAllState = () => {
    resetProducingMusicState();
    resetCoverGenHeightState();
    resetSheduleArrayState();
    resetPrevDataState();
    resetLastScheduleTimeState();
    resetBlobAudioState();
    resetCoversState();
    resetSelectedCoverState();
    resetSelectedCoverPromptState();
    resetTrackIdsState();
    resetTracksInfoState();
    resetMusicalInstrumentsState();
    resetCanPostMusicState();
    resetTrack0();
    resetTrack1();
    resetTrack2();
    resetTrack3();
    resetTrack4();
    // router.push("/produce");
  };

  useEffect(() => {
    resetAllState();
  }, []);

  return <></>;
}

export default Reset;
