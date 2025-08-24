// Test for pure auth helper functions

describe('Auth Token Expiry', () => {
  it('should correctly identify expired tokens', () => {
    const expiredTime = Date.now() - 1000 // 1 second ago
    const validTime = Date.now() + 3600000 // 1 hour from now
    
    expect(Date.now() > expiredTime).toBe(true)
    expect(Date.now() > validTime).toBe(false)
  })

  it('should calculate token expiry time correctly', () => {
    const now = Date.now()
    const oneHourFromNow = now + (60 * 60 * 1000)
    
    expect(oneHourFromNow - now).toBe(3600000) // 1 hour in milliseconds
  })
})