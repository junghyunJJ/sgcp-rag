import { z } from "zod";

// 회원가입 스키마
export const signupSchema = z
  .object({
    email: z.string().email("유효한 이메일 주소를 입력해주세요.").min(1, "이메일은 필수 입력 항목입니다."),
    password: z.string().min(6, "비밀번호는 최소 6자 이상이어야 합니다."),
    confirmPassword: z.string().min(1, "비밀번호 확인은 필수 입력 항목입니다."),
  })
  .refine((data) => data.password === data.confirmPassword, {
    message: "비밀번호가 일치하지 않습니다.",
    path: ["confirmPassword"],
  });

// 로그인 스키마
export const loginSchema = z.object({
  email: z.string().email("유효한 이메일 주소를 입력해주세요.").min(1, "이메일은 필수 입력 항목입니다."),
  password: z.string().min(1, "비밀번호를 입력해주세요."),
});

// 스키마에서 타입 추출
export type SignupFormValues = z.infer<typeof signupSchema>;
export type LoginFormValues = z.infer<typeof loginSchema>; 