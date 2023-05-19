import {
  QueryClient,
  QueryKey,
  useMutation,
  useQuery,
} from "@tanstack/react-query";
import { AxiosResponse } from "axios";
import { CustomQueryHookReturnType, UseQueryOptionsType } from "../types/query";

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: Infinity,
    },
  },
});

export const customInvalidate = (queryKey: QueryKey, options = {}) =>
  queryClient.invalidateQueries(queryKey, options);

export const useQueryResult = (
  queryKey: QueryKey,
  queryFn: (...args: any[]) => Promise<AxiosResponse<any, any>>,
  options?: UseQueryOptionsType,
): CustomQueryHookReturnType => {
  const query = useQuery(queryKey, queryFn, options);
  //주석 console.log(query.data?.data);
  return [query.data?.data, query.refetch, query];
};

export const prefetchingQuery = async (
  queryKey: QueryKey,
  queryFn: (...args: any[]) => Promise<AxiosResponse<any, any>>,
) => {
  await queryClient.prefetchQuery({
    queryKey,
    queryFn,
  });
};

export const useCustomMutate = (
  queryKey: QueryKey,
  mutationFn: (...args: any[]) => Promise<AxiosResponse<any, any>>,
) => {
  const mutation = useMutation(mutationFn, {
    onSuccess: () => {
      customInvalidate(queryKey);
    },
  });
  return mutation.mutate;
};

export default queryClient;
