// Learn more: https://github.com/testing-library/jest-dom
import '@testing-library/jest-dom'

// Polyfill for TextEncoder/TextDecoder
import { TextEncoder, TextDecoder } from 'util'
global.TextEncoder = TextEncoder
global.TextDecoder = TextDecoder

Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: jest.fn().mockImplementation((query) => ({
    matches: false,
    media: query,
    onchange: null,
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    addListener: jest.fn(),
    removeListener: jest.fn(),
    dispatchEvent: jest.fn(),
  })),
})

// Mock next-auth
jest.mock('next-auth/react', () => ({
  useSession: jest.fn(),
  signIn: jest.fn(),
  signOut: jest.fn(),
}), { virtual: true })

// Mock next/navigation
jest.mock('next/navigation', () => ({
  useRouter: () => ({
    push: jest.fn(),
    replace: jest.fn(),
    prefetch: jest.fn(),
    back: jest.fn(),
  }),
  usePathname: () => '/',
  useSearchParams: () => new URLSearchParams(),
}))

// Mock ESM-only markdown packages for Jest's CommonJS runtime.
jest.mock('react-markdown', () => {
  const React = require('react')
  return {
    __esModule: true,
    default: ({ children }) => {
      const blocks = String(children || '')
        .split(/\n+/)
        .map((line) =>
          line
            .replace(/^#{1,6}\s*/, '')
            .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
            .trim()
        )
        .filter(Boolean)

      return React.createElement(
        'div',
        null,
        blocks.map((block, index) => React.createElement('p', { key: index }, block))
      )
    },
  }
})

jest.mock('remark-gfm', () => jest.fn())
jest.mock('remark-breaks', () => jest.fn())
