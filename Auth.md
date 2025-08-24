# Authentication Architecture Documentation

This document provides comprehensive guidance for the authentication implementation in LangConnect Client using NextAuth.js with Supabase.

## Overview

LangConnect Client uses NextAuth.js with JWT strategy for authentication, integrated with Supabase as the authentication provider. The system implements automatic token refresh to maintain user sessions seamlessly.

## Architecture

### Token Flow

```
┌─────────────┐     ┌──────────────┐     ┌───────────┐
│   Browser   │────▶│   NextAuth   │────▶│ Supabase  │
│             │◀────│   (JWT)      │◀────│   Auth    │
└─────────────┘     └──────────────┘     └───────────┘
     │                      │
     │ httpOnly cookie     │ refresh token
     │ (encrypted JWT)     │ stored in JWT
     │                      │
     ▼                      ▼
 Only accessToken      Auto refresh when
 exposed to client     accessToken expires
```

### Key Components

1. **NextAuth Configuration** (`/next-connect-ui/src/lib/auth.ts`)
   - JWT strategy for session management
   - Credentials provider for email/password authentication
   - Callbacks for JWT and session handling

2. **Backend Auth API** (`/langconnect/api/auth.py`)
   - Supabase integration for user management
   - Token generation and validation
   - Refresh token endpoint

3. **Client Hooks** (`/next-connect-ui/src/hooks/use-auth.ts`)
   - React hooks for authentication state
   - Login/logout/register functions
   - Session management

## Implementation Details

### 1. Token Storage Strategy

**Current Implementation:**
- Access token: Stored in NextAuth JWT and exposed to client session
- Refresh token: Currently not stored (needs implementation)

**Recommended Implementation:**
```typescript
// In JWT callback
async jwt({ token, user }) {
  if (user) {
    return {
      ...token,
      accessToken: user.accessToken,
      refreshToken: user.refreshToken,
      accessTokenExpires: Date.now() + (60 * 60 * 1000), // 1 hour
    }
  }
  
  // Return previous token if not expired
  if (Date.now() < token.accessTokenExpires) {
    return token
  }
  
  // Refresh the token
  return await refreshAccessToken(token)
}
```

### 2. Automatic Token Refresh

```typescript
async function refreshAccessToken(token) {
  try {
    const response = await fetch(`${API_URL}/auth/refresh`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        refresh_token: token.refreshToken
      })
    })
    
    const refreshed = await response.json()
    
    if (!response.ok) {
      throw new Error(refreshed.detail || 'Failed to refresh token')
    }
    
    return {
      ...token,
      accessToken: refreshed.access_token,
      refreshToken: refreshed.refresh_token,
      accessTokenExpires: Date.now() + (60 * 60 * 1000), // 1 hour
    }
  } catch (error) {
    console.error('Error refreshing access token', error)
    
    return {
      ...token,
      error: "RefreshAccessTokenError",
    }
  }
}
```

### 3. Session Callback Updates

```typescript
async session({ session, token }) {
  if (token && session.user) {
    session.user.id = token.id as string
    session.user.email = token.email as string
    session.user.name = token.name as string
    session.user.accessToken = token.accessToken as string
    // Check for refresh errors
    if (token.error) {
      session.error = token.error
    }
  }
  
  return session
}
```

### 4. TypeScript Type Definitions

```typescript
// next-auth.d.ts
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
```

## Security Considerations

### 1. Token Security
- **Never expose refresh tokens to the client**: Only store in server-side JWT
- **Use httpOnly cookies**: NextAuth automatically handles this
- **Implement token rotation**: Always get new refresh token when refreshing

### 2. Error Handling
- **Handle refresh failures gracefully**: Redirect to login on refresh failure
- **Log authentication events**: Monitor for suspicious activity
- **Implement rate limiting**: Prevent brute force attacks

### 3. Best Practices
- **Short-lived access tokens**: Supabase default is 1 hour
- **Secure token transmission**: Always use HTTPS
- **Validate tokens server-side**: Don't trust client-side validation

## Testing Strategy

### 1. Unit Tests
```typescript
// Test token refresh function
describe('refreshAccessToken', () => {
  it('should refresh valid token', async () => {
    // Mock API response
    const mockToken = {
      refreshToken: 'valid-refresh-token',
      accessToken: 'old-access-token',
    }
    
    const refreshed = await refreshAccessToken(mockToken)
    
    expect(refreshed.accessToken).not.toBe('old-access-token')
    expect(refreshed.refreshToken).toBeDefined()
  })
  
  it('should handle refresh failure', async () => {
    // Mock API error
    const mockToken = {
      refreshToken: 'invalid-refresh-token',
    }
    
    const result = await refreshAccessToken(mockToken)
    
    expect(result.error).toBe('RefreshAccessTokenError')
  })
})
```

### 2. Integration Tests
- Test login flow with token storage
- Test automatic refresh on expired token
- Test logout and token cleanup

### 3. E2E Tests
- Test full authentication flow
- Test protected route access
- Test token expiration handling

## Migration Guide

To implement refresh token support:

1. Update NextAuth configuration with new callbacks
2. Update TypeScript types
3. Implement refresh token function
4. Update axios interceptor to handle token refresh
5. Test thoroughly with expired tokens

## Monitoring and Logging

Implement logging for:
- Successful logins
- Token refresh attempts
- Authentication failures
- Suspicious patterns

## Future Enhancements

1. **Implement refresh token rotation**
   - Generate new refresh token on each use
   - Invalidate old refresh tokens

2. **Add session extension**
   - Allow users to extend session before expiry
   - Implement "Remember me" functionality

3. **Multi-device session management**
   - Track active sessions per user
   - Allow users to revoke sessions

4. **Enhanced security features**
   - Two-factor authentication
   - Device fingerprinting
   - Anomaly detection