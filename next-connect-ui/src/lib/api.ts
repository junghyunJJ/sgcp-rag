/**
 * 백엔드 API와 통신하기 위한 유틸리티 함수
 */

import type { NextRequest } from "next/server"
import axios, { AxiosRequestConfig, AxiosResponse } from "axios"
import api from "./axios"

type FetchOptions = {
  method?: "GET" | "POST" | "PUT" | "DELETE" | "PATCH"
  headers?: Record<string, string>
  body?: any
  cache?: RequestCache
  next?: NextRequest
}

/**
 * API 요청을 보내는 기본 함수
 */
export async function fetchAPI<T = any>(endpoint: string, options: FetchOptions = {}): Promise<T> {
  const { method = "GET", headers = {}, body, ...customConfig } = options

  const config: AxiosRequestConfig = {
    method,
    headers: {
      ...headers,
    },
    ...customConfig,
  }

  if (body) {
    config.data = body
  }

  try {
    const response: AxiosResponse<T> = await api(endpoint, config)
    return response.data
  } catch (error) {
    if (axios.isAxiosError(error)) {
      console.error(`API Error for ${endpoint}:`, {
        status: error.response?.status,
        data: error.response?.data,
        message: error.message
      })
      if (error.response?.data?.message) {
        return Promise.reject(new Error(error.response.data.message))
      }
      if (error.response?.data?.detail) {
        return Promise.reject(new Error(error.response.data.detail))
      }
    }
    return Promise.reject(new Error("서버 오류가 발생했습니다."))
  }
}

/**
 * 서버 컴포넌트에서 사용할 API 요청 함수
 */
export async function serverFetchAPI<T = any>(endpoint: string, options: FetchOptions = {}): Promise<T> {
  return fetchAPI<T>(endpoint, {
    ...options,
    headers: {
      ...options.headers,
      "Content-Type": "application/json",
    },
  })
}

/**
 * FormData 업로드를 위한 API 요청 함수
 */
export async function uploadFormData<T = any>(endpoint: string, formData: FormData): Promise<T> {
  return fetchAPI<T>(endpoint, {
    method: "POST",
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    body: formData,
  })
}

