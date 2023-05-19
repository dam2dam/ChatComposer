import React, { useEffect, useState } from "react";
import { Tab } from "@headlessui/react";
import Link from "next/link";
import {
  IconBrandDeezer,
  IconLogout,
  IconMusic,
  IconPlaylist,
  IconPlus,
  IconSearch,
  IconUserCircle,
} from "@tabler/icons-react";
import { useRouter } from "next/router";

function Sidebar() {
  const routes = ["/", "/sequencer", "/chat"];
  const [opacities, setOpacities] = useState([
    "opacity-50",
    "opacity-50",
    "opacity-50",
  ]);
  const { pathname } = useRouter();
  useEffect(() => {
    routes.forEach((route, index) => {
      if (pathname.startsWith(route)) {
        setOpacities([
          ...opacities.map((e, i) => {
            if (i === index) {
              return "opacity-90";
            } else return "opacity-50";
          }),
        ]);
      }
    });
  }, [pathname]);
  return (
    <div className="mp_sidebar absolute w-[100px] h-full flex flex-col justify-between items-center px-0 py-[25px] rounded-[20px_0px_0px_20px] left-0 top-0">
      <div className="sidebar_logo ">
        <img
          className="sidebar_logo_img w-[50px] mb-[30px]"
          src="https://cdn-icons-png.flaticon.com/512/3293/3293810.png"
        />
      </div>
      <Tab.Group vertical>
        <Tab.List className="sidebar_menu flex flex-col gap-[30px] flex-[1]">
          <Tab>
            <Link href="/" className={opacities[0]}>
              <IconPlaylist className="sidebar_menu_image w-5 transition-[0.2s] duration-[ease-in-out] hover:cursor-pointer hover:scale-125 text-white" />
            </Link>
          </Tab>
          <Tab>
            <Link href="/sequencer" className={opacities[1]}>
              <IconMusic className="sidebar_menu_image w-5 transition-[0.2s] duration-[ease-in-out] hover:cursor-pointer hover:scale-125 text-white" />
            </Link>
          </Tab>
          <Tab>
            <Link href="/chat" className={opacities[2]}>
              <IconUserCircle className="sidebar_menu_image w-5 transition-[0.2s] duration-[ease-in-out] hover:cursor-pointer hover:scale-125 text-white" />
            </Link>
          </Tab>
          <Tab>
            <Link href="/produce" className={opacities[2]}>
              <IconPlus className="sidebar_menu_image w-5 transition-[0.2s] duration-[ease-in-out] hover:cursor-pointer hover:scale-125 text-white" />
            </Link>
          </Tab>
          <Tab>
            <Link href="/chat" className={opacities[3]}>
              <IconSearch className="sidebar_menu_image w-5 transition-[0.2s] duration-[ease-in-out] hover:cursor-pointer hover:scale-125 text-white" />
            </Link>
          </Tab>
        </Tab.List>
      </Tab.Group>
      <div className="sidebar_logout">
        <IconLogout className="sidebar_logout_login @apply w-[30px] transition-[0.2s] duration-[ease-in-out] hover:cursor-pointer hover:scale-125 text-white opacity-50 hover:opacity-90" />
      </div>
    </div>
  );
}

export default Sidebar;
