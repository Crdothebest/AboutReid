import axios from 'axios';

export const apiClient = axios.create({
  baseURL: '/api',
  timeout: 20000,
});

apiClient.interceptors.response.use(
  (resp) => resp,
  (error) => {
    return Promise.reject(error);
  }
);


