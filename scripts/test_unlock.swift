// test_unlock — Keychain-backed key vault with Touch ID gate.
//
// Design note: macOS CLI binaries that aren't signed with a proper
// `keychain-access-groups` entitlement cannot attach a SecAccessControl
// (biometry ACL) to keychain items — SecItemAdd returns -34018
// errSecMissingEntitlement. To make this work without a signing setup,
// storage and biometric gating are decoupled:
//
//   1. Keychain stores a plain generic password (no ACL) accessible
//      whenever the keychain is unlocked (same as ssh-agent session).
//   2. Before every fetch, this helper explicitly calls
//      LAContext.evaluatePolicy(deviceOwnerAuthenticationWithBiometrics)
//      and refuses to return the key unless the user authenticates.
//      Falls back to deviceOwnerAuthentication (biometric OR passcode).
//   3. Each invocation = fresh process = fresh LAContext = fresh prompt.
//
// The weaker property vs. entitled-signed binaries: the keychain item
// itself is readable by other same-user processes that know the service
// name and bypass this helper. For our threat model (prevent accidental
// reads by autoresearch / claude subprocesses, not defeat targeted
// bypass by a human with shell access), this is the right tradeoff.
//
// Modes:
//   test_unlock --set-key   read base64 key from stdin, store (no prompt)
//   test_unlock             gate on Touch ID, then fetch + print key
//
// Compile:
//   swiftc scripts/test_unlock.swift \
//       -framework LocalAuthentication -framework Security \
//       -o scripts/test_unlock
//   codesign --force --sign - scripts/test_unlock   # ad-hoc, optional

import Foundation
import LocalAuthentication
import Security

let SERVICE  = "autoresearch-test-aes-key"
let ACCOUNT  = "default"
let REASON   = "Decrypt autoresearch test dataset"

var stderr = FileHandle.standardError

func eprint(_ s: String) {
    stderr.write((s + "\n").data(using: .utf8) ?? Data())
}

func setKey(_ keyBase64: String) -> Bool {
    guard let keyData = Data(base64Encoded: keyBase64) else {
        eprint("test_unlock: stdin is not valid base64")
        return false
    }

    // Wipe any existing item first so we idempotently overwrite.
    let deleteQuery: [CFString: Any] = [
        kSecClass:       kSecClassGenericPassword,
        kSecAttrService: SERVICE,
        kSecAttrAccount: ACCOUNT,
    ]
    SecItemDelete(deleteQuery as CFDictionary)

    // Plain generic password; no ACL. Using the legacy file keychain
    // (file-based login.keychain) by NOT setting
    // kSecUseDataProtectionKeychain — file keychain is reachable from
    // unsigned CLI tools. AccessibleWhenUnlockedThisDeviceOnly is
    // advisory here (ignored by the file keychain) but harmless.
    let addQuery: [CFString: Any] = [
        kSecClass:             kSecClassGenericPassword,
        kSecAttrService:       SERVICE,
        kSecAttrAccount:       ACCOUNT,
        kSecValueData:         keyData,
        kSecAttrAccessible:    kSecAttrAccessibleWhenUnlockedThisDeviceOnly,
    ]
    let status = SecItemAdd(addQuery as CFDictionary, nil)
    if status != errSecSuccess {
        eprint("test_unlock: SecItemAdd failed (OSStatus \(status))")
        return false
    }
    return true
}

/// Prompt Touch ID (or device passcode) and block until the user
/// authenticates. Returns true on success. Separate from keychain
/// fetch so the gate is always applied explicitly, regardless of
/// SecItem access control.
func authenticate() -> Bool {
    let ctx = LAContext()
    ctx.localizedReason = REASON
    ctx.touchIDAuthenticationAllowableReuseDuration = 0

    // Prefer biometric; fall back to device passcode automatically via
    // .deviceOwnerAuthentication. The policy is "either biometric or
    // passcode, whichever is available and the user provides."
    var policyErr: NSError?
    let policy: LAPolicy = .deviceOwnerAuthentication
    guard ctx.canEvaluatePolicy(policy, error: &policyErr) else {
        eprint("test_unlock: authentication policy unavailable: "
               + (policyErr?.localizedDescription ?? "unknown"))
        return false
    }

    let sem = DispatchSemaphore(value: 0)
    var success = false
    var authError: Error?
    ctx.evaluatePolicy(policy, localizedReason: REASON) { ok, err in
        success = ok
        authError = err
        sem.signal()
    }
    sem.wait()

    if !success {
        if let e = authError {
            eprint("test_unlock: authentication failed: \(e.localizedDescription)")
        } else {
            eprint("test_unlock: authentication canceled")
        }
        return false
    }
    return true
}

func fetchKey() -> String? {
    let query: [CFString: Any] = [
        kSecClass:          kSecClassGenericPassword,
        kSecAttrService:    SERVICE,
        kSecAttrAccount:    ACCOUNT,
        kSecReturnData:     true,
        kSecMatchLimit:     kSecMatchLimitOne,
    ]
    var item: CFTypeRef?
    let status = SecItemCopyMatching(query as CFDictionary, &item)
    if status != errSecSuccess {
        eprint("test_unlock: SecItemCopyMatching failed (OSStatus \(status))")
        return nil
    }
    guard let data = item as? Data else {
        eprint("test_unlock: item had no data payload")
        return nil
    }
    return data.base64EncodedString()
}

let args = CommandLine.arguments

if args.contains("--help") || args.contains("-h") {
    print("Usage: test_unlock [--set-key | --help]")
    print("  default      prompt Touch ID (or passcode), print base64 key to stdout")
    print("  --set-key    read base64 key from stdin, store in Keychain (no prompt)")
    exit(0)
}

if args.contains("--set-key") {
    guard let line = readLine() else {
        eprint("test_unlock: stdin empty (expected base64 key)")
        exit(1)
    }
    exit(setKey(line.trimmingCharacters(in: .whitespacesAndNewlines)) ? 0 : 1)
}

// Fetch path: authenticate first, then read keychain.
if !authenticate() {
    exit(1)
}
if let b64 = fetchKey() {
    print(b64)
    exit(0)
}
exit(1)
