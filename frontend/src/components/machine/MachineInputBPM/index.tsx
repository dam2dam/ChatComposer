import { useRecoilState } from "recoil";
import { Transport } from "tone";
import machineTempoAtom from "../../../lib/store/machineTempoAtom";
import InputRange from "../../shared/InputRange";

const MachineInputBPM = ({}) => {
  const [machineTempoState, setMachineTempoState] =
    useRecoilState(machineTempoAtom);

  const handleUpdateTempo = (e: React.ChangeEvent<HTMLInputElement>) => {
    let newTempo = Number(e.target.value);
    if (newTempo > 480) newTempo = 480;
    else if (newTempo < 10) newTempo = 10;
    // Transport.bpm.value = newTempo * 4; //16분음표
    Transport.bpm.value = newTempo; //4분음표
    setMachineTempoState(newTempo);
  };

  return (
    <InputRange
      label="BPM"
      min={10}
      max={480}
      value={machineTempoState}
      handleOnChange={(e) => handleUpdateTempo(e)}
    />
  );
};

export default MachineInputBPM;
