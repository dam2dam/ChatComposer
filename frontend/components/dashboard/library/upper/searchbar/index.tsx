import React from "react";

function Searchbar() {
  return (
    <div className="searchbar w-[600px] h-10 flex items-center ml-5 rounded-[50px] bg-pink-800/40">
      <img
        className="searchbar_img w-5 ml-5"
        src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAABmJLR0QA/wD/AP+gvaeTAAABxElEQVRIidWUv27UQBCHv1m7ubsUpCf0EY9wXh/HKyBFugaq0PAEFKmQ8gKIhj9NhHSRDvEKR7y+gheAPtCeghRSIMs7FLeGBBzbR4IQv2bs0fzm2x2vF/53SV3SObcpIjtAoqpbIX0cRVFeluXMWnvyx4DFYnHXe78H9GsNImfAkyRJ5msDQvP9VR858t4fxnH8MTTeLopiIiKpiHhVfWytfdcZ4JzbBN4CfWPM0+Fw+LrO4Jy7DzwSka9FUdwbj8dfmgDmB2k1876IHF3WHMBaewA4Vd2I43inbQfm3LMF8N4ftpmAaYhJZ4Cq3gSoZt6kXq9X1Ww1FnJxB9pWXGm5XNYe70aAiHwOcbvNNBgMqppPnQHGGAdQFMWkzaSqkxBdZ0BZljMRORORNBzFWuV5/oDVxz0VkTdtgAuzzPN8DOyrqgEcMI2i6AOA9/52WHkS3p+PRqNXawEAnHN3RGRPVTcu8Zx676fGmIcAqvoyTdMXnQEA8/n8RviJEuBWSB8DOTCz1p44595X9U2QzsftV1VXxrnUgbX22bUBukKuBOgCuTKgDWLqLesp3LA/Vy3y7Tr6/qYsy3azLNv9K83/mb4D+s23Z1Qya+gAAAAASUVORK5CYII="
      />
      <p className="searchbar_p text-gray-700 text-xs ml-2.5">
        Search for songs, artist. playlists...
      </p>
    </div>
  );
}

export default Searchbar;
