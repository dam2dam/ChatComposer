import { atom } from "recoil";

const machineBeatsCount = atom({
  key: "beats",
  default: 64,
});

export default machineBeatsCount;
