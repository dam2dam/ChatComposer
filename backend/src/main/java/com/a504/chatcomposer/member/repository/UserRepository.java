package com.a504.chatcomposer.member.repository;

import com.a504.chatcomposer.member.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface UserRepository extends JpaRepository<User, Long> {
    User findByUserId(String userId);

    Optional<User> findByUserSeq(Long userSeq);
}
