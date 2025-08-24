import { refreshAccessToken } from '../../lib/auth'

// Mock fetch globally
global.fetch = jest.fn()

describe('refreshAccessToken', () => {
  beforeEach(() => {
    jest.clearAllMocks()
  })

  it('should successfully refresh a valid token', async () => {
    const mockToken = {
      id: 'user-123',
      email: 'test@example.com',
      name: 'Test User',
      accessToken: 'old-access-token',
      refreshToken: 'valid-refresh-token',
      accessTokenExpires: Date.now() - 1000, // Expired
    }

    const mockResponse = {
      access_token: 'new-access-token',
      refresh_token: 'new-refresh-token',
      user_id: 'user-123',
      email: 'test@example.com',
    }

    ;(global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    })

    const result = await refreshAccessToken(mockToken)

    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/auth/refresh'),
      expect.objectContaining({
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          refresh_token: mockToken.refreshToken,
        }),
      })
    )

    expect(result).toEqual({
      ...mockToken,
      accessToken: 'new-access-token',
      refreshToken: 'new-refresh-token',
      accessTokenExpires: expect.any(Number),
    })

    expect(result.accessTokenExpires).toBeGreaterThan(Date.now())
  })

  it('should handle refresh failure', async () => {
    const mockToken = {
      id: 'user-123',
      email: 'test@example.com',
      name: 'Test User',
      accessToken: 'old-access-token',
      refreshToken: 'invalid-refresh-token',
      accessTokenExpires: Date.now() - 1000,
    }

    ;(global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: false,
      json: async () => ({ detail: 'Invalid refresh token' }),
    })

    const result = await refreshAccessToken(mockToken)

    expect(result).toEqual({
      ...mockToken,
      error: 'RefreshAccessTokenError',
    })
  })
})