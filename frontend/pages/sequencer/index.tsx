import dynamic from "next/dynamic";
import Image from "next/image";
import { Suspense } from "react";
import Chat from "../chat";
// import AudioTest from "../../components/AudioTest";
const Machine = dynamic(
  () => import("../../src/components/machine/MachineView"),
  {
    ssr: false,
  },
);
const NotesView = dynamic(
  () => import("../../src/components/notes/NotesView/NotesView"),
  {
    ssr: false,
  },
);
const AudioTest = dynamic(() => import("../../components/AudioTest"), {
  ssr: false,
});
const Sequencer = () => {
  return (
    <div className="">
      <AudioTest />
      <Chat />
      <Suspense>
        <Machine />
        <NotesView />
      </Suspense>
      {/* 시퀀서 앱 출처 표기 */}
      <div className="self-center p-2 w-18 h-18 bg-black text-white flex">
        powered by farmeroy
        <a href="https://github.com/farmeroy/synth-app">
          <Image
            alt="Link to Github"
            width={28}
            height={28}
            src="/github-mark-white.svg"
          />
        </a>
      </div>
    </div>
  );
};

export default Sequencer;
