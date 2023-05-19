import axios from "axios";
import React, { useEffect, useState } from "react";
import { useRecoilCallback } from "recoil";
import noteRowActiveBeatsAtom from "../src/lib/store/noteRowActiveBeatsAtom";
import Link from "next/link";
import notesAtom from "../src/lib/store/notesAtom";
import machineBeatsCount from "../src/lib/store/machineBeatsCount";

const Chat = () => {
  const ChatApi = axios.create();

  const [message, setMessage] = useState("");
  const [input, setInput] = useState("");

  const prevData = [
    { role: "user", content: "classicMusic" },
    {
      role: "assistant",
      content:
        "G4-1/4-0, G4-1/4-0.5, A4-1/4-1, A4-1/4-1.5, B4-1/4-2, B4-1/4-2.5, A4-1/2-3, G4-1/2-3.5, G4-1/4-4, A4-1/4-4.5, A4-1/4-5, B4-1/4-5.5, B4-1/4-6, A4-1/2-6.5, G4-1/2-7.",
    },
  ];

  const [notes, setNotes] = useState([]);

  const setAllDone = useRecoilCallback(({ set }) => () => {
    let scaleSet = new Set();
    let lastIndex = 1;

    // 배열의 각 요소에 대해
    notes.forEach((e) => {
      // 해당 id의 atom을 가져와서
      const note = e[0].toLowerCase();
      const length = e[1];
      const index = e[2];
      const activeNotes = noteRowActiveBeatsAtom(note);

      // active를 true로 설정
      set(activeNotes, (activeNotes) => {
        const state = [...activeNotes];
        state[index] = !activeNotes[index];
        return state;
      });

      //악기를 사용하도록 집합에 추가
      scaleSet.add(note);
      lastIndex = Math.max(lastIndex, index + length);
    });

    const scaleArray = Array.from(scaleSet);
    set(notesAtom, (notes) => {
      const state = [...notes];
      scaleArray.forEach((e) => {
        const index = state.findIndex((note) => note.note === e);
        state[index] = { ...state[index], isActive: true };
      });
      return state;
    });

    set(machineBeatsCount, () => {
      // return maxIndex + 1 * 4;//16분음표
      //4분 음표
      // return maxIndex + 1;
      return lastIndex;
    });
  });

  useEffect(() => {
    if (!notes.length) return;
    setAllDone();
  }, [notes]);

  const [baseUrl, setBaseUrl] = useState("https://k8a504.p.ssafy.io");

  useEffect(() => {
    if (window.location.host.startsWith("localhost")) {
      setBaseUrl("http://localhost:3000");
    }
  }, []);

  return (
    <div>
      <input value={input} onChange={(e) => setInput(e.target.value)}></input>
      <button
        onClick={() => {
          ChatApi.post(`${baseUrl}/api/chatgpt`, {
            message: input,
            prevData,
          }).then((res) => {
            setMessage(JSON.stringify(res.data));
            setNotes(res.data.noteInfo);
          });
        }}
      >
        전송하기
      </button>
      <Link href="/sequencer">악보로 이동하기</Link>
      <div>{message}</div>
    </div>
  );
};

export default Chat;
