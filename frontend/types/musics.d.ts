import { GenreMap, TagMap } from "../utils/GenreMap";
import { InstrumentsMap } from "../utils/InstrumentsMap";

export interface AlbumCoverType {
  member: {
    member_id?: number;
    memberId?: number;
    nickname: string;
  };
  music_id: number;
  title: string;
  genre: string;
  tags: string[];
  favorite_count: number;
  is_my_favorite: "y" | "n";
  created_at: string;
  cover_source: string;
}

export type GenreType = keyof typeof GenreMap;
export type InstrumentType = keyof typeof InstrumentsMap;
export type TagType = keyof typeof TagMap;
export type DrumKitType =
  | "OpenHats"
  | "ClosedHats"
  | "PedalHats"
  | "Crash"
  | "Clap"
  | "Toms"
  | "Snare"
  | "Kick"
  | "SubKick";

declare module ".*mp3";

export interface MusicType {
  member: {
    member_id?: number;
    memberId?: number;
    nickname: string;
  };
  music_id: number;
  title: string;
  genre: string;
  tags: string[];
  favorite_count: number;
  created_at: string;
  cover_source: string;
  is_my_favorite: string;
}

export type MusicListType = MusicType[];

export interface TrackType {
  track_id: number;
  midi_description: string; //음악을 재생하기 위한 midi 악보
  musical_instrument: string; //악기의 종류
}

export interface PromptType {
  prompt_id: number;
  request_description: string; //사용자의 요청
  response_description: string; //chatGPT의 응답
  transfer_date: number; //요청시간
}

export interface MusicDetailType extends MusicType {
  description: String; //음악에 대한 설명(GPT가 만들수도 있음, 음악 이어하기 했을 때에는 강제로 출처가 표기되게)
  beat: String;

  //아래의 두 배열의 같은 인덱스는 1:1로 연결될 트랙과 프롬프트이다.
  tracks: TrackType[];
  prompts: PromptType[];

  //미디
  music_source: string; //미디 원음 wav의 url

  //리퓨전
  mixed_music_request: string; //리퓨전에 쓰인 프롬프트
  mixed_music_source: string; //리퓨전으로 합성된 wav의 url

  //앨범
  cover_request: string; //앨범 커버 얻을 때 쓴 키워드
  cover_source: string; //앨범 커버 url
  favorite_count: string; //좋아요 수
  is_my_favorite: string; // 좋아요 여부
}
