import { AtomEffect, atom, atomFamily, useSetRecoilState } from "recoil";
import { recoilPersist } from "recoil-persist";
import { Todo } from "../types/todos";
import { GenreType, InstrumentType, MusicType, TagType } from "../types/musics";

//Next.js에서 persistAtom을 쓰기 위한 구성
//https://stackoverflow.com/questions/68110629/nextjs-react-recoil-persist-values-in-local-storage-initial-page-load-in-wrong
const ssrCompletedState = atom({
  key: "SsrCompleted",
  default: false,
});

export const useSsrComplectedState = () => {
  const setSsrCompleted = useSetRecoilState(ssrCompletedState);
  return () => setSsrCompleted(true);
};

const { persistAtom } = recoilPersist();

export const persistAtomEffect = <T>(param: Parameters<AtomEffect<T>>[0]) => {
  param.getPromise(ssrCompletedState).then(() => persistAtom(param));
};

//atom({key:, default:})로 새로운 아톰을 만들 수 있다.
// 이때 key는 각 아톰을 구별하는 고유한 식별자이다.
// default는 initial state를 의미한다.

export const todoListState = atom({
  key: "Todos",
  default: [] as Todo[],
  // effects_UNSTABLE: [persistAtom],
  effects_UNSTABLE: [persistAtomEffect],
});

export const authTokenState = atom({
  key: "AuthTokenState",
  default: "",
  effects_UNSTABLE: [persistAtomEffect],
});

export const loginState = atom<boolean>({
  key: "loginState",
  default: false,
  effects_UNSTABLE: [persistAtom],
});

export const trackAtomFamily = atomFamily({
  key: "trackAtomFamily",
  default: (id) => ({
    id,
    midi_description: "",
    musical_instrument: null as number,
    request_description: "",
    response_description: "",
    transfer_date: "",
  }),
});

export const producingMusicState = atom({
  key: "producingMusicState",
  default: {
    title: null as string, //음악 제목
    tags: null as TagType[],
    description: null as string, //음악에 대한 설명(GPT가 만들수도 있음, 음악 이어하기 했을 때에는 강제로 출처가 표기되게)
    genre: null as number,
    beat: null as number,
    created_at: null as string,
    //미디
    music_source: null as string, //미디 원음 wav의 url

    //리퓨전
    mixed_music_request: null as string, //리퓨전에 쓰인 프롬프트
    mixed_music_source: null as string, //리퓨전으로 합성된 wav의 url

    //앨범
    cover_request: null as string, //앨범 커버 얻을 때 쓴 키워드
    cover_source: null as string, //앨범 커버 url
  },
});

export const CoverGenHeightState = atom({
  key: "CoverGenHeightState",
  default: "h-0 opacity-0",
});

export const sheduleArrayState = atom({
  key: "SheduleArrayState",
  default: [],
});

export const prevDataState = atom({
  key: "PrevDataState",
  default: [],
});

export const lastScheduleTimeState = atom({
  key: "LastScheduleTimeState",
  default: Number.MAX_SAFE_INTEGER,
});

export const blobAudioState = atom({
  key: "BlobAudioState",
  default: null as null | Blob,
});

export const coversState = atom({
  key: "CoversState",
  default: [],
});

export const selectedCoverState = atom({
  key: "SelectedCoverState",
  default: "",
});

export const selectedCoverPromptState = atom({
  key: "SelectedCoverPromptState",
  default: "",
});

export const trackIdsState = atom({
  key: "TrackIdsState",
  default: [] as object[][],
});

export const tracksInfoState = atom({
  key: "TracksInfoState",
  default: [] as object[],
});

export const musicalInstrumentsState = atom({
  key: "MusicalInstrumentsState",
  default: [] as string[],
});

export const canPostMusicState = atom({
  key: "CanPostMusicState",
  default: false,
});

export const selectedMusicState = atom({
  key: "selectedMusicState",
  default: null as MusicType,
});

export const genreDataState = atom({
  key: "genreDataState",
  default: [],
});

export const selectedGenreState = atom({
  key: "selectedGenreState",
  default: null as GenreType,
});
