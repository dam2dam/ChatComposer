package com.a504.chatcomposer.member.service;

import com.a504.chatcomposer.member.entity.User;
import com.a504.chatcomposer.member.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
public class UserService {
    private final UserRepository userRepository;

    public User getUser(String userId) {
        return userRepository.findByUserId(userId);
    }

    public Long getUserSeq(String userId) {
        return userRepository.findByUserId(userId).getUserSeq();
    }
}
