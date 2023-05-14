package com.a504.chatcomposer.user.service;

import com.a504.chatcomposer.user.entity.User;
import com.a504.chatcomposer.user.repository.UserRepository;
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
