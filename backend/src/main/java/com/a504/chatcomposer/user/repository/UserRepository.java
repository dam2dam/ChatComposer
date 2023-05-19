package com.a504.chatcomposer.user.repository;

import java.util.Optional;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import com.a504.chatcomposer.user.entity.User;

@Repository
public interface UserRepository extends JpaRepository<User, Long>, UserCustomRepository {

	User findByUserId(String userId);

	Optional<User> findByUserSeq(Long userSeq);

	User getReferenceByUserId(String username);
}
