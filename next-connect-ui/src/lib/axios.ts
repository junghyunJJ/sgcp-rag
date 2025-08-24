import axios from 'axios';

const getApiUrl = () => {
  if (typeof window === 'undefined') {
    // Server-side
    return process.env.API_URL || "http://localhost:8080"
  }
  // Client-side
  return process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080"
}

export const API_URL = getApiUrl()

// axios 인스턴스 생성
const api = axios.create({
  baseURL: API_URL,
})

// Request interceptor removed - no authentication needed

api.interceptors.response.use(
  (response: any) => {
    // Handle 204 No Content responses
    if (response.status === 204) {
      response.data = { success: true };
    }
    return response;
  },
  (error: any) => {
    return Promise.reject(error);
  }
);


export default api;
