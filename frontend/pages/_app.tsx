import type { AppProps } from "next/app";
import "./app.css";
import Head from "next/head";
import { RecoilRoot } from "recoil";
import { QueryClientProvider } from "@tanstack/react-query";
import queryClient from "../services";
import Layout from "../components/Layout";
// import { SessionProvider } from "next-auth/react";
import { ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";

export default function App({ Component, pageProps }: AppProps) {
  return (
    // <SessionProvider session={pageProps.session}>
    <>
      <Head>
        <title>ChatComposer</title>
      </Head>
      <QueryClientProvider client={queryClient}>
        <ToastContainer />
        <RecoilRoot>
          <Layout>
            <Component {...pageProps} />
          </Layout>
        </RecoilRoot>
      </QueryClientProvider>
    </>
    // </SessionProvider>
  );
}
