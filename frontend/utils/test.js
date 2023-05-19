let time = 0;
let arr = [];

const duration = "-1/4-";

for (let i = 0; i < 120; i++) {
  const current = i % 16;
  // const houseMap = {
  //   OpenHats: [2, 5, 10, 13],
  //   Tom: [2, 10],
  //   Snare: [4, 12],
  //   Kick: [0, 4, 8, 12],
  //   ClosedHats: Array.from({ length: 16 }, (e, i) => i),
  // };
  // const map = {
  //   //Dubstep
  //   OpenHats: [],
  //   ClosedHats: [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 14, 15],
  //   PedalHats: [],
  //   Crash: [6, 13],
  //   Clap: [],
  //   Toms: [],
  //   Snare: [5, 12],
  //   Kick: [0, 3, 8, 11, 13, 15],
  //   SubKick: [],
  // };
  const map = {
    //Oldschool
    OpenHats: [2, 10],
    ClosedHats: Array.from({ length: 8 }, (e, i) => {
      if (i !== 2) return i * 2;
      else return 16;
    }),
    PedalHats: [],
    Crash: [],
    Clap: [5, 13, 15],
    Toms: [],
    Snare: [4, 12],
    Kick: [2, 10],
    SubKick: [0, 7, 8],
  };

  Object.entries(map).forEach((e) => {
    const name = e[0];
    const value = e[1];
    // if (value.includes(current)) arr.push(name + duration + time);
    if (value.includes(current)) arr.push([name, 0.5, time]);
  });

  time += 0.5;
}
//주석 console.log("4비트 South HipHop");
//주석 console.log(JSON.stringify(arr));

/*
OpenHats: "OpenHH.mp3",
ClosedHats: "ClosedHH.mp3",
PedalHats: "PedalHH.mp3",
Crash: "Crash.mp3",
Clap: "Clap.mp3",
Toms: "Tom.mp3",
Snare: "Snare.mp3",
Kick: "Kick.mp3",
SubKick: "SubKick.mp3",

*/
