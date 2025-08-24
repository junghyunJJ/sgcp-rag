import "next-auth"

declare module "next-auth" {
  interface Session {
    user: User
    error?: string
  }

  interface User {
    id: string
    email: string
    name: string
    accessToken: string
    refreshToken?: string
  }
}

declare module "next-auth/jwt" {
  interface JWT {
    id: string
    email: string
    name: string
    accessToken: string
    refreshToken: string
    accessTokenExpires: number
    error?: string
  }
}