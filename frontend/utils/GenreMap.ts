import { TagType } from "./../types/musics.d";
import { GenreType } from "../types/musics";

export const GenreMap = {
  POP: "팝",
  HIP_HOP: "힙합",
  ROCK: "락",
  SOUL: "소울",
  REGGAE: "레게",
  COUNTRY: "컨트리",
  FUNK: "펑크",
  FOLK: "포크송",
  MIDDLE_EASTERN: "중동음악",
  JAZZ: "재즈",
  DISCO: "디스코",
  CLASSICAL: "클래식",
  ELECTRONIC: "일렉트로니카",
  BLUES: "블루스",
  NEW_AGE: "뉴에이지",
  VOCAL: "성악",
  CHRISTIAN: "찬송가",
  SKA: "레게",
  TRADITIONAL: "고대 음악",
  INDEPENDENT: "인디 음악",
};

export const GenreMapEntries = Object.entries(GenreMap) as [
  GenreType,
  string,
][];

export const TagMap = {
  Happy: "행복한",
  Sad: "슬픈",
  Energetic: "활기찬",
  Calm: "차분한",
  Romantic: "로맨틱한",
  Melancholic: "멜랑콜리한",
  Uplifting: "격려하는",
  Mysterious: "신비로운",
  Exciting: "흥분된",
  Peaceful: "평화로운",
  Dreamy: "몽환적인",
  Sentimental: "감성적인",
  Powerful: "강력한",
  Joyful: "기쁜",
  Reflective: "반성하는",
  Playful: "장난스러운",
  Nostalgic: "향수를 자아내는",
  Intense: "강렬한",
  Serene: "고요한",
  Inspiring: "영감을 주는",
};

export const TagMapEntries = Object.entries(TagMap) as [TagType, string][];
