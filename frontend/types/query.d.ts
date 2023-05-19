export type CustomQueryHookReturnType<TData = any> = [
  TData,
  <TPageData>(
    options?: RefetchOptions & RefetchQueryFilters<TPageData>,
  ) => Promise<QueryObserverResult<AxiosResponse<any, any>, unknown>>,
  UseQueryResult<TData, TError>,
];

export type UseQueryOptionsType = Omit<
  UseQueryOptions<TQueryFnData, TError, TData, TQueryKey>,
  "initialData" | "queryFn" | "queryKey"
> & {
  initialData?: () => undefined;
};

export type CustomQueryHookType<TParams = unknown, TData = any> = (
  params?: TParams,
  options?: UseQueryOptionsType,
) => CustomQueryHookReturnType<TData>;

export type AxiosQueryFuntion = () => () => Promise<AxiosResponse<any, any>>;
