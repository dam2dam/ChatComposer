import { atomFamily } from "recoil";

//실행되는 노트가 true입니다. 기본값은 모두 false입니다.
const noteRowActiveBeatsAtom = atomFamily({
  key: "noteRowActiveBeatsAtom",
  default: [false, false, false, false],
});

export default noteRowActiveBeatsAtom;
