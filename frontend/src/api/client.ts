import axios from 'axios';

export const apiClient = axios.create({
  baseURL: '/api',
  timeout: 10000, // 减少到10秒
  headers: {
    'Content-Type': 'application/json',
  },
});

apiClient.interceptors.response.use(
  (resp) => resp,
  (error) => {
    console.error('API请求失败:', error.message);
    return Promise.reject(error);
  }
);

// 添加请求拦截器
apiClient.interceptors.request.use(
  (config) => {
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);


